# HuggingFace Scripts

Utilities for pushing data/models to and pulling them from the EdgeDeID HuggingFace collection.

**Collection:** `https://huggingface.co/EdgeDeID`  
**Dataset repo:** `EdgeDeID/synthetic-dataset`  
**Model repos:** `EdgeDeID/Qwen3-0.6B-Base_SFT`, `EdgeDeID/Qwen3-1.7B-Base_SFT`, `EdgeDeID/Qwen3-0.6B-GRPO-KL`

---

## Authentication

Each team member uses a personal HF token — no tokens in the repo.

**One-time setup (local):**
1. Go to `https://huggingface.co/settings/tokens` → New token → Read + Write
2. Run `huggingface-cli login` and paste your token

Token is saved to `~/.huggingface/token`. All scripts pick it up automatically.

**Colab / Kaggle:** create a `.env` file at the repo root (it's gitignored):
```
HF_TOKEN=hf_xxxxxxxxxxxx
```

---

## Scripts

### `load_from_hf.py` — load data into notebooks

```python
from scripts.load_from_hf import load_dataset, load_teachers

# Single teacher (pseudolabeled)
df = load_dataset("deepseek", pseudolabeled=True)

# All three teachers
data = load_teachers(pseudolabeled=True)   # {"deepseek": df, "haiku": df, "hermes": df}

# Pre-download everything for offline use
from scripts.load_from_hf import download_all
download_all()
```

First call downloads from HF and caches to `./data/`. Subsequent calls load from cache.

### `push_to_hf.py` — push data to HF

Scans `data/{deepseek,haiku,hermes}/` and uploads all CSVs in a single commit.

```bash
python scripts/push_to_hf.py
```

### `fetch_hf_info.py` — snapshot collection metadata

Prints metadata (downloads, tags, file list) for all models and datasets, and saves to `hf_collection_info.json`.

```bash
python scripts/fetch_hf_info.py
```

### `verify_hf_collection.py` — health-check all repos

Confirms every model and dataset is reachable, lists files, loads configs and tokenizers.

```bash
python scripts/verify_hf_collection.py
```

---

## Dataset structure on HF

```
EdgeDeID/synthetic-dataset/
├── deepseek/
│   ├── output_1_deepseek.csv … output_6_deepseek.csv
│   └── deepseek_pseudolabeled.csv
├── haiku/
│   ├── output_1_haiku.csv … output_7_haiku.csv
│   └── haiku_pseudolabeled.csv
└── hermes/
    ├── output_1_hermes.csv … output_17_hermes.csv
    └── hermes_pseudolabeled.csv
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Repository Not Found` | Check repo exists and you're authenticated: `huggingface-cli whoami` |
| `ModuleNotFoundError: scripts` | Add repo root to path: `sys.path.insert(0, "..")` |
| `SSL EOF error` | Network issue — retry or check VPN/proxy |
