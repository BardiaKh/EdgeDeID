# Hugging Face Dataset Pipeline

Smart automation to push/pull EdgeDeID synthetic data to/from Hugging Face Hub **without manual copying**.

**Dataset repo:** `https://huggingface.co/datasets/EdgeDeID/synthetic-dataset`

---

## 🚀 Quick Start (No Manual Copying!)

### 1️⃣ Push data to HF (one-time setup)

```bash
pip install huggingface_hub tqdm
huggingface-cli login  # Paste your HF token

# From EdgeDeID root:
python scripts/push_to_hf.py
```

✓ Reads from `/data/` directly, uploads CSVs to HF.

### 2️⃣ Use data in Colab/Kaggle (auto-caching)

```python
# Install (one-time in Colab)
!pip install huggingface_hub

# Load data
from scripts.load_from_hf import load_dataset, load_teachers
import pandas as pd

# Load one teacher
df = load_dataset("deepseek", pseudolabeled=True)

# Load all teachers
data = load_teachers(pseudolabeled=True)  # {"deepseek": df, "haiku": df, ...}
df_all = pd.concat(data.values(), ignore_index=True)
```

✓ Data automatically downloads once, then caches locally.

---

## Dataset Structure

```
EdgeDeID/synthetic-dataset/  (HF dataset repo)
├── deepseek/                     (6 raw outputs + 1 pseudolabeled)
│   ├── output_1_deepseek.csv
│   ├── output_2_deepseek.csv
│   ├── ...output_6_deepseek.csv
│   └── deepseek_pseudolabeled.csv
├── haiku/                        (7 raw outputs + 1 pseudolabeled)
│   ├── output_1_haiku.csv
│   ├── ...output_7_haiku.csv
│   └── haiku_pseudolabeled.csv
└── hermes/                       (17 raw outputs + 1 pseudolabeled)
    ├── output_1_hermes.csv
    ├── ...output_17_hermes.csv
    └── hermes_pseudolabeled.csv
```

**Total:** raw outputs + pseudolabeled -> CSV files

---

## Part 1: Push to HF (scripts/push_to_hf.py)

### What it does

Automatically scans `/data/{deepseek,haiku,hermes}/` and uploads all CSVs to HF in one commit.

### Usage

```bash
python scripts/push_to_hf.py
```

**Output:**
```
🚀 EdgeDeID → Hugging Face Dataset Push
Found 6 files in deepseek/
Found 7 files in haiku/
Found 17 files in hermes/
📤 Uploading 33 files...
✓ Successfully uploaded!
  Commit: https://huggingface.co/datasets/EdgeDeID/synthetic-dataset/tree/...
```

### Adding new data later

After generating new files (e.g., `output_8_haiku.csv`):

```bash
python scripts/push_to_hf.py  # Uploads only new files
```

---

## Part 2: Load from HF (scripts/load_from_hf.py)

### Basic examples

**Load one teacher:**
```python
from scripts.load_from_hf import load_dataset
df = load_dataset("deepseek", pseudolabeled=True)
```

**Load all teachers:**
```python
from scripts.load_from_hf import load_teachers
data = load_teachers(pseudolabeled=True)  # Dict: {teacher: df}
```

**Load raw outputs (not pseudolabeled):**
```python
df_raw = load_dataset("hermes", pseudolabeled=False)
```

**Pre-download everything (offline later):**
```python
from scripts.load_from_hf import download_all
download_all()  # ~10 min, then works offline
```

### How it works

- **First call:** Downloads from HF (~5–10 min), caches to `./data/`
- **Later calls:** Loads from cache (~instant)
- **After cache:** Works without internet

### In your notebooks

**combine.ipynb:**
```python
import sys
sys.path.insert(0, "..")

from scripts.load_from_hf import load_teachers

data = load_teachers(pseudolabeled=True)
df_deepseek = data["deepseek"]
df_haiku = data["haiku"]
df_hermes = data["hermes"]

# Rest of your code continues...
```

**pseudolabel.ipynb:**
```python
import sys
sys.path.insert(0, "..")

from scripts.load_from_hf import load_dataset

dataset_name = "hermes"
df_raw = load_dataset(dataset_name, pseudolabeled=False)

# Your pseudolabeling code...
```

---

## Troubleshooting

### Push fails: "Cannot access EdgeDeID/synthetic-dataset"

1. Verify repo exists: https://huggingface.co/datasets/EdgeDeID/synthetic-dataset
2. Check your token: `huggingface-cli whoami`
3. Re-login: `huggingface-cli login`

### Load fails: "ModuleNotFoundError"

```python
import sys
sys.path.insert(0, "..")  # Add EdgeDeID root to path
from scripts.load_from_hf import load_dataset
```

### Dataset not visible after push

- Refresh HF hub page (`Ctrl+Shift+R`)
- Wait few minutes for indexing

---

## Advanced: Custom loading

See docstrings in `scripts/load_from_hf.py`:

```python
# Suppress verbose output
df = load_dataset("haiku", pseudolabeled=True, verbose=False)

# Load multiple times with progress
df_all = load_teachers(pseudolabeled=False, verbose=True)
```

---

## References

- [scripts/push_to_hf.py](scripts/push_to_hf.py) — Full push implementation
