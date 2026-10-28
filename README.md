# EdgeDeID

**A 0.6B clinical de-identifier trained without real patient data.**

[![Paper](https://img.shields.io/badge/arXiv-2610.33839-b31b1b.svg)](https://arxiv.org/abs/2610.33839)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EdgeDeID%2Fsynthetic--dataset-yellow)](https://huggingface.co/datasets/EdgeDeID/synthetic-dataset)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3--0.6B--Base__SFT-yellow)](https://huggingface.co/EdgeDeID/Qwen3-0.6B-Base_SFT)

- **Paper:** [arXiv:2610.33839](https://arxiv.org/abs/2610.33839) ([PDF](https://arxiv.org/pdf/2610.33839))
- **Dataset:** [`EdgeDeID/synthetic-dataset`](https://huggingface.co/datasets/EdgeDeID/synthetic-dataset)
- **Model:** [`EdgeDeID/Qwen3-0.6B-Base_SFT`](https://huggingface.co/EdgeDeID/Qwen3-0.6B-Base_SFT)
- **All released artifacts:** [huggingface.co/EdgeDeID](https://huggingface.co/EdgeDeID)

Sending a clinical note to a hosted language model discloses the information
that de-identification is meant to protect. EdgeDeID is a small decoder-only
model (Qwen3-0.6B) that runs locally, even on CPU, and is trained only on
synthetic text. Three LLM teachers write synthetic notes, a single labeler
re-annotates them into one labeling convention, and the model learns to emit a
typed XML schema of the identifiers it finds (names, ages, dates, places,
addresses, phone numbers, emails, ids, URLs, devices).

From the paper's abstract:

- On a 100-note synthetic in-domain benchmark with manually verified
  annotations, EdgeDeID reaches **0.994 micro-F1** and leaves no missed
  identifier in **99% of notes**. The strongest of eight off-the-shelf systems
  reaches 0.857 and 12%.
- Quantized, it runs on CPU in 615 MB.
- Of three additions to supervised training, only a verifier adapter helps. It
  separates complete from incomplete outputs of other systems at 0.94 AUROC.
  Reinforcement learning with a recall-weighted reward and a rule-based
  post-processing layer leave in-domain results unchanged.
- On MIMIC ICU nursing notes, F1 falls to 0.48. The errors follow two
  properties the synthetic notes lack: telegraphic style and low identifier
  density. Three hand-written in-context examples recover 0.08 F1 without
  annotated target data.

---

## Released artifacts

| Artifact | Hugging Face | What it is |
|---|---|---|
| Dataset | [`EdgeDeID/synthetic-dataset`](https://huggingface.co/datasets/EdgeDeID/synthetic-dataset) | Synthetic training and test notes with `<phi_entities>` labels (`train.csv`, `test.csv`; columns `original`, `anonymized`, `examples`) |
| EdgeDeID (SFT, 0.6B) | [`EdgeDeID/Qwen3-0.6B-Base_SFT`](https://huggingface.co/EdgeDeID/Qwen3-0.6B-Base_SFT) | The paper's de-identifier |
| Labeler (SFT, 1.7B) | [`EdgeDeID/Qwen3-1.7B-Base_SFT`](https://huggingface.co/EdgeDeID/Qwen3-1.7B-Base_SFT) | The model that re-annotates the teachers' notes into one convention |
| GRPO with KL anchor | [`EdgeDeID/Qwen3-0.6B-GRPO-KL`](https://huggingface.co/EdgeDeID/Qwen3-0.6B-GRPO-KL) | The reinforcement-learning variant from the paper's ablation |

---

## Quick start

```bash
git clone https://github.com/BardiaKh/EdgeDeID.git
cd EdgeDeID
uv sync            # or: pip install -e .   (Python >= 3.10)
```

The model is a plain causal LM with a flat prompt (no chat template). It
generates a `<phi_entities>` block, which the library maps back onto the note:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edgedeid.data.xml_utils import get_entity_positions, mask_text
from edgedeid.prompts import SYSTEM_PROMPT

REPO = "EdgeDeID/Qwen3-0.6B-Base_SFT"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, device_map="auto")

note = open("demo/sample_note.txt").read()
prompt = (
    f"<instructions>\n{SYSTEM_PROMPT}\n</instructions>\n\n"
    f"<note>\n{note.strip()}\n</note>\n\n"
    "<phi_entities>\n"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs, max_new_tokens=1024, do_sample=False,
    stop_strings=["</phi_entities>"], tokenizer=tok,
)
xml = "<phi_entities>\n" + tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
xml = xml.split("</phi_entities>")[0] + "</phi_entities>"

positions = get_entity_positions(note, xml)
print(mask_text(note, positions, mask_char="*"))
```

Decoding is greedy. The prompt above is the same one the evaluation code
builds (`edgedeid.eval.inference._build_prompt`).

Load the dataset with `datasets`:

```python
from datasets import load_dataset

ds = load_dataset(
    "EdgeDeID/synthetic-dataset",
    data_files={"train": "train.csv", "test": "test.csv"},
)
```

---

## Repository layout

```
edgedeid/                  Installable library
  prompts.py               The SYSTEM_PROMPT (PHI tagging schema)
  data/                    preprocessing, augmentations, XML utilities
  eval/                    metrics, inference, the 100-note benchmark loader, MIMIC scoring
  qc/                      verifier adapter and the rule-based QC tiers
  rl/                      GRPO: config, rewards, evaluation
  training/                expand_tokenizer (adds the XML tag tokens to Qwen3)
  baselines/               wrappers for the off-the-shelf systems compared in the paper
notebooks/
  1_data/                  data_generation, pseudolabel, combine
  2_train/                 sft, grpo
  3_eval/                  baselines, ablation, qc_ablation, en_eval, synth_eval
scripts/                   CLI entry points: vLLM serving, QC verifier training and eval,
                           MIMIC scoring, latency, Hugging Face utilities (see scripts/README.md)
configs/                   Prompt configs for the three teachers
demo/                      gradio_demo.ipynb, sample_note.txt
evals/                     Evaluation data and result tables
artifacts/mimic/           Tracked MIMIC evaluation outputs
RUNBOOK.md                 Detailed command-by-command runbook for the paper's experiments
```

---

## Reproducing the pipeline

The training recipe has five steps. Each notebook states its inputs and outputs
in its first cells.

1. **Generate synthetic notes** with the three teachers (Claude Haiku,
   DeepSeek, Hermes-405B): `notebooks/1_data/data_generation.ipynb`.
   Set `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` for the
   teachers you want to regenerate. Prompt configs are in `configs/`.
2. **Train the labeler**: `notebooks/2_train/sft.ipynb` on `Qwen/Qwen3-1.7B-Base`.
3. **Relabel** every teacher's notes into one convention with the labeler,
   served by vLLM: `notebooks/1_data/pseudolabel.ipynb`.
4. **Combine and augment** (label-preserving date, name and phone-format
   augmentations) into `train.csv` and `test.csv`:
   `notebooks/1_data/combine.ipynb` with `PSEUDO_LABEL = True`.
5. **Train the final model**: `notebooks/2_train/sft.ipynb` on `Qwen/Qwen3-0.6B-Base`.
   Training is completion-only SFT with TRL, bf16, Liger kernels and a
   warmup-stable-decay schedule (`lr = 5e-5`, max length 8192).

The released dataset is the output of step 4, so you can start at step 5.

The additions studied in the paper are reproducible too:

- **GRPO**: `notebooks/2_train/grpo.ipynb` (`edgedeid/rl/`).
- **Verifier adapter**: `scripts/build_qc_dataset.py`, `scripts/train_qc_sft.py`,
  `scripts/train_qc_grpo.py`, `scripts/eval_qc_verifier.py` (`edgedeid/qc/`).
- **Baselines** (Presidio, GLiNER, BERT/RoBERTa de-identifiers, scispaCy, Stanford
  de-id, OpenMed, Qwen and others): `notebooks/3_eval/baselines.ipynb`.
- **Latency**: `scripts/benchmark_latency.py`.

`RUNBOOK.md` lists the exact commands, in order.

### Running on CPU

Convert a checkpoint with [llama.cpp](https://github.com/ggml-org/llama.cpp) and
quantize it:

```bash
python convert_hf_to_gguf.py /path/to/checkpoint --outfile edgedeid-BF16.gguf --outtype bf16
./build/bin/llama-quantize edgedeid-BF16.gguf edgedeid-Q4_K_M.gguf Q4_K_M
```

The GGUF files are not hosted on the Hub.

---

## Evaluation data

| Benchmark | Location | Notes |
|---|---|---|
| 100-note synthetic in-domain benchmark | `evals/ext_eval/LLM_Eval.xlsx` | Loaded by `edgedeid.eval.benchmark` |
| MIMIC ICU nursing notes | `evals/ext_eval/physionet_deid/`, `mimic_data*.csv` | From the PhysioNet de-identification gold standard (Open Data Commons Attribution License); see `evals/ext_eval/physionet_deid/SOURCE.md` |

Score a run on the MIMIC notes with `scripts/score_mimic.py`.

---

## Data provenance

All training data is synthetic: the notes are written by LLMs and contain no
real patient text. The MIMIC evaluation notes are real and de-identified. Their
original source and license are in `evals/ext_eval/physionet_deid/SOURCE.md`.

---

## Citation

```bibtex
@article{daneshgar2026edgedeid,
  title   = {EdgeDeID: A 0.6B Clinical De-identifier Trained Without Real Patient Data},
  author  = {Daneshgar, Sina and Namazi, Raya and Khosravi, Amirali and Rouzrokh, Pouria and Khosravi, Bardia},
  journal = {arXiv preprint arXiv:2610.33839},
  year    = {2026}
}
```
