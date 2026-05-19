"""
Shared pipeline utilities for synthetic medical report generation across
DeepSeek, Haiku (Anthropic batch), and Hermes (vLLM/RadPrompter) backends.
"""

import os
import re
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Shared data-generation helpers
# ---------------------------------------------------------------------------

def generate_random_features(
    num_features: int,
    all_features: Optional[list[str]] = None,
) -> dict[str, str]:
    """Return a random PHI feature set and report-length label.

    Args:
        num_features: Total number of PHI categories to include (≥1).
        all_features: Ordered list of feature names; first entry is always
            included. Defaults to the full 10-category set.
    """
    if all_features is None:
        all_features = [
            "name", "age", "date", "hospital", "address",
            "phone", "email", "id", "url", "device",
        ]

    features = ["name"]
    remaining = random.sample(
        all_features[1:], min(num_features - 1, len(all_features) - 1)
    )
    features.extend(remaining)

    if num_features > 8:
        weights = [0.1, 0.1, 0.8]
    elif num_features > 5:
        weights = [0.2, 0.2, 0.6]
    else:
        weights = [0.1, 0.3, 0.6]

    length = random.choices(["short", "medium", "long"], weights=weights, k=1)[0]
    return {"report_length": length, "features_to_include": ", ".join(features)}


def generate_random_inputs(
    num_inputs: int,
    all_features: Optional[list[str]] = None,
    num_features_range: tuple[int, int] = (5, 11),
) -> list[dict[str, str]]:
    """Generate a list of random PHI-feature configurations.

    Args:
        num_inputs: Number of samples to generate.
        all_features: Feature pool forwarded to generate_random_features.
        num_features_range: (min, max) inclusive range for feature count.
    """
    return [
        generate_random_features(
            random.randint(*num_features_range),
            all_features=all_features,
        )
        for _ in range(num_inputs)
    ]


# ---------------------------------------------------------------------------
# Shared output helpers
# ---------------------------------------------------------------------------

def parse_report_response(text: str) -> tuple[str, str]:
    """Extract raw and anonymized report strings from a model response."""
    raw = re.search(r"([\s\S]*?)</raw_report>", text).group(1)
    anon = re.search(r"<anonymized_report>([\s\S]*?)</anonymized_report>", text).group(1)
    return raw, anon


def save_records(records: dict, output_path: str) -> pd.DataFrame:
    """Write records dict to CSV and return the resulting DataFrame."""
    df = pd.DataFrame.from_dict(records, orient="index")
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDataGenerator(ABC):
    """Abstract base for synthetic medical report generation pipelines."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    @abstractmethod
    def run(self, criteria: list[dict[str, str]]) -> pd.DataFrame:
        """Run the generation pipeline and persist results to output_path."""

    def save_log(self, log_path: str) -> None:  # noqa: ARG002
        """No-op for backends that don't produce a RadPrompter generation log."""


# ---------------------------------------------------------------------------
# DeepSeek — sequential OpenAI-compatible API
# ---------------------------------------------------------------------------

class DeepSeekGenerator(BaseDataGenerator):
    """Sequential generator using the DeepSeek chat API."""

    def __init__(
        self,
        output_path: str,
        system_prompt: str,
        user_prompt_template: str,
        model: str = "deepseek-chat",
    ):
        super().__init__(output_path)
        from openai import OpenAI
        from tqdm import tqdm
        self._tqdm = tqdm
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def run(self, criteria: list[dict[str, str]]) -> pd.DataFrame:
        records = {}
        for i, criterion in enumerate(self._tqdm(criteria)):
            user_msg = (
                self.user_prompt_template
                .replace("{{report_length}}", criterion["report_length"])
                .replace("{{features_to_include}}", criterion["features_to_include"])
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
            )
            raw, anon = parse_report_response(response.choices[0].message.content)
            records[f"message_{i}"] = {
                "criterion": criterion,
                "raw_report": raw,
                "anonymized_report": anon,
            }
        return save_records(records, self.output_path)


# ---------------------------------------------------------------------------
# Haiku — Anthropic Messages Batch API with prompt caching
# ---------------------------------------------------------------------------

class HaikuGenerator(BaseDataGenerator):
    """Batch generator using the Anthropic Messages Batch API with prompt caching."""

    def __init__(
        self,
        output_path: str,
        system_prompt: str,
        cached_prompt: str,
        variable_prompt_template: str,
        response_template: str,
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        super().__init__(output_path)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.system_prompt = system_prompt
        self.cached_prompt = cached_prompt
        self.variable_prompt_template = variable_prompt_template
        self.response_template = response_template
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_requests(
        self, criteria: list[dict[str, str]]
    ) -> tuple[list, dict]:
        batch_requests, records = [], {}
        for i, criterion in enumerate(criteria):
            variables = (
                self.variable_prompt_template
                .replace("{{report_length}}", criterion["report_length"])
                .replace("{{features_to_include}}", criterion["features_to_include"])
            )
            batch_requests.append({
                "custom_id": f"message_{i}_turn_1",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": self.system_prompt,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.cached_prompt,
                                    "cache_control": {"type": "ephemeral"},
                                },
                                {"type": "text", "text": variables},
                            ],
                        },
                        {"role": "assistant", "content": self.response_template},
                    ],
                },
            })
            records[f"message_{i}"] = {
                "criterion": criterion,
                "raw_report": None,
                "anonymized_report": None,
            }
        return batch_requests, records

    def _poll_until_done(self, batch_id: str) -> None:
        while True:
            update = self.client.beta.messages.batches.retrieve(batch_id)
            if update.processing_status == "ended":
                print(f"Status: {update.processing_status}")
                print(f"Created: {update.created_at}")
                print(f"Ended: {update.ended_at}")
                print(f"Expires: {update.expires_at}")
                break
            time.sleep(1)

    def run(self, criteria: list[dict[str, str]]) -> pd.DataFrame:
        batch_requests, records = self._build_requests(criteria)
        response = self.client.beta.messages.batches.create(requests=batch_requests)
        batch_id = response.id
        print(f"Batch {batch_id} Summary:")
        self._poll_until_done(batch_id)
        for result in self.client.beta.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                record_id = f"message_{result.custom_id.split('_')[1]}"
                raw, anon = parse_report_response(result.result.message.content[0].text)
                records[record_id]["raw_report"] = f"<raw_report>{raw}</raw_report>"
                records[record_id]["anonymized_report"] = f"<anonymized_report>{anon}</anonymized_report>"
        return save_records(records, self.output_path)


# ---------------------------------------------------------------------------
# Hermes — RadPrompter with vLLM-compatible backend
# ---------------------------------------------------------------------------

class HermesGenerator(BaseDataGenerator):
    """Generator using RadPrompter with a vLLM-compatible OpenRouter backend."""

    def __init__(
        self,
        output_path: str,
        prompt_path: str,
        model: str = "nousresearch/hermes-3-llama-3.1-405b:free",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        seed: int = 29301,
        concurrency: int = 1,
    ):
        super().__init__(output_path)
        from radprompter import Prompt, RadPrompter, vLLMClient
        self._RadPrompter = RadPrompter
        self.prompt = Prompt(prompt_path)
        self.client = vLLMClient(
            model=model,
            base_url=base_url,
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=temperature,
            seed=seed,
        )
        self.concurrency = concurrency
        self._engine = None

    def run(self, criteria: list[dict[str, str]]) -> None:
        """Run RadPrompter; output is written directly to output_path."""
        self._engine = self._RadPrompter(
            client=self.client,
            prompt=self.prompt,
            concurrency=self.concurrency,
            output_file=self.output_path,
        )
        self._engine(criteria)

    def save_log(self, log_path: str) -> None:
        """Save the RadPrompter generation log (call after run())."""
        if self._engine is None:
            raise RuntimeError("Call run() before save_log().")
        self._engine.save_log(log_path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pipeline(
    backend: str,
    output_path: str,
) -> tuple["BaseDataGenerator", dict]:
    """Instantiate the right generator and return kwargs for generate_random_inputs.

    Args:
        backend: One of ``"deepseek"``, ``"haiku"``, or ``"hermes"``.
        output_path: Destination CSV path for generated records.

    Returns:
        ``(generator, generate_kwargs)`` — pass generate_kwargs directly to
        :func:`generate_random_inputs` so model-specific feature restrictions
        are applied automatically.
    """
    from radprompter import Prompt

    if backend == "deepseek":
        prompt = Prompt("../data_prep_config/prompt-anthropic.toml")
        generator = DeepSeekGenerator(
            output_path=output_path,
            system_prompt=prompt.system_prompt,
            user_prompt_template=prompt.user_prompts[0],
        )
        return generator, {}

    if backend == "haiku":
        prompt = Prompt("../data_prep_config/prompt-anthropic-rad-2.toml")
        user_prompt = prompt.user_prompts[0]
        cached = user_prompt.rsplit("</anonymized_report>", 1)[0] + "</anonymized_report>"
        variables = user_prompt.rsplit("</anonymized_report>", 1)[1]
        generator = HaikuGenerator(
            output_path=output_path,
            system_prompt=prompt.system_prompt,
            cached_prompt=cached,
            variable_prompt_template=variables,
            response_template=prompt.response_templates[0].strip(),
        )
        return generator, {}

    if backend == "hermes":
        generator = HermesGenerator(
            output_path=output_path,
            prompt_path="../data_prep_config/prompt.toml",
        )
        return generator, {"all_features": ["name"], "num_features_range": (2, 11)}

    raise ValueError(f"Unknown backend {backend!r}. Choose 'deepseek', 'haiku', or 'hermes'.")
