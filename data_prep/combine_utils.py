"""
Utility functions for combine.ipynb:
multi-source data loading, response cleaning, and report manipulation.
"""

import re
from glob import glob
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_hermes(pseudo_label: bool) -> pd.DataFrame:
    pattern = "../data/hermes/hermes_pseudolabeled.csv" if pseudo_label else "../data/hermes/output_*_hermes.csv"
    frames = []
    for path in glob(pattern):
        df = pd.read_csv(path)
        df.drop(index=df[df["features_to_include"].isna()].index, inplace=True)
        df.drop(index=df[df["default_response_1"].isna()].index, inplace=True)
        df["model"] = "Hermes"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    try:
        out.drop(columns=["index"], inplace=True)
        faulty_indices = [18765, 19717, 21618, 21362, 21524, 21507, 22274, 22506]
        out.drop(index=faulty_indices, inplace=True, errors="ignore")
    except Exception:
        pass  # pseudolabeled files don't have the index column or faulty indices
    out.reset_index(inplace=True, drop=True)
    return out


def _load_haiku(pseudo_label: bool) -> pd.DataFrame:
    pattern = "../data/haiku/haiku_pseudolabeled.csv" if pseudo_label else "../data/haiku/output_*_haiku.csv"
    frames = []
    for path in glob(pattern):
        df = pd.read_csv(path)
        df["model"] = "Haiku"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    try:
        out.rename(columns={"raw_report": "default_response_0", "anonymized_report": "default_response_1"}, inplace=True)
        out["report_length"] = out["criterion"].apply(lambda x: eval(x)["report_length"] if pd.notna(x) else None)
        out["features_to_include"] = out["criterion"].apply(lambda x: eval(x)["features_to_include"] if pd.notna(x) else None)
        out.drop(columns=["criterion"], inplace=True)
    except Exception:
        out["features_to_include"] = out["features_to_include"].astype(str)
    out.reset_index(inplace=True, drop=True)
    return out[["default_response_0", "default_response_1", "report_length", "features_to_include", "model"]]


def _load_deepseek(pseudo_label: bool) -> pd.DataFrame:
    pattern = "../data/deepseek/deepseek_pseudolabeled.csv" if pseudo_label else "../data/deepseek/output_*_deepseek.csv"
    frames = []
    for path in glob(pattern):
        df = pd.read_csv(path)
        df["model"] = "DeepSeek"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    try:
        out.rename(columns={"raw_report": "default_response_0", "anonymized_report": "default_response_1"}, inplace=True)
        out["report_length"] = out["criterion"].apply(lambda x: eval(x)["report_length"] if pd.notna(x) else None)
        out["features_to_include"] = out["criterion"].apply(lambda x: eval(x)["features_to_include"] if pd.notna(x) else None)
        out.drop(columns=["criterion"], inplace=True)
    except Exception:
        out["features_to_include"] = out["features_to_include"].astype(str)
    out.reset_index(inplace=True, drop=True)
    return out[["default_response_0", "default_response_1", "report_length", "features_to_include", "model"]]


_LOADERS = {"hermes": _load_hermes, "haiku": _load_haiku, "deepseek": _load_deepseek}


def load_combined_dataset(data_source: List[str], pseudo_label: bool) -> pd.DataFrame:
    """Load and merge datasets for all requested sources.

    Args:
        data_source: Names of sources to include (e.g. ``["hermes", "haiku"]``).
        pseudo_label: If True, load pseudolabeled CSVs; otherwise load raw generation outputs.
    """
    counts = {}
    frames = []
    for name, loader in _LOADERS.items():
        df = loader(pseudo_label)
        counts[name] = len(df)
        if len(df) > 0 and name in data_source:
            frames.append(df)

    print(f"Haiku rows:   {counts['haiku']}")
    print(f"Hermes rows:  {counts['hermes']}")
    print(f"DeepSeek rows:{counts['deepseek']}")
    return pd.concat(frames) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

def clean_combined_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Strip XML wrapper tags, drop null/empty/duplicate response rows in-place."""
    words_to_remove = [
        "<anonymized_report>", "</anonymized_report>", "<raw_report>", "</raw_report>",
        "<anonymized_report/>", "<raw_report/>", "</anonymized_report",
    ]
    for word in words_to_remove:
        df["default_response_0"] = df["default_response_0"].str.replace(word, "", regex=False)
        df["default_response_1"] = df["default_response_1"].str.replace(word, "", regex=False)

    df["default_response_0"] = df["default_response_0"].str.strip()
    df["default_response_1"] = df["default_response_1"].str.strip()

    df.drop(index=df[df["default_response_0"].isna()].index, inplace=True)
    df.drop(index=df[df["default_response_1"].isna()].index, inplace=True)
    df.drop(index=df[df["default_response_0"] == ""].index, inplace=True)
    df.drop(index=df[df["default_response_1"] == ""].index, inplace=True)
    df.drop_duplicates(subset=["default_response_1"], inplace=True)
    df.drop_duplicates(subset=["default_response_0"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------

def verify_tag_presence(text: str, tags: List[str]) -> List[str]:
    """Return the subset of tags not found as open/close pairs in text."""
    return [tag for tag in tags if f"<{tag}" not in text or f"</{tag}>" not in text]


def update_id_attribute(text: str, prefix: str) -> str:
    """Prefix the id attribute of every <name> and <place> tag with the given string."""
    pattern = r'<(name|place)\s+[^>]*?id="([^"]*)"[^>]*>(.*?)</(name|place)>'
    for match in re.finditer(pattern, text):
        id_attr = match.group(2)
        text = text.replace(f'id="{id_attr}"', f'id="{prefix}{id_attr}"')
    return text
