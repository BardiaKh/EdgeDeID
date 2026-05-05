# Hugging Face Dataset Pipeline

Smart automation to push/pull EdgeDeID synthetic data to/from Hugging Face Hub **without manual copying**.

**Dataset repo:** `https://huggingface.co/datasets/EdgeDeID/synthetic-dataset`

---

## Secure Token Setup

**Each team member** generates a personal HF token — zero tokens in the repo.

### Local Development (Recommended)

1. **Generate token** at https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: `EdgeDeID` (or your name)
   - Permissions: **Read + Write**

2. **Store locally** (one-time):
   ```bash
   huggingface-cli login
   ```
   Paste your token. It's saved to `~/.huggingface/token` (NOT in repo).

3. **Scripts automatically use it** — no config needed!

✓ Each person has their own token  
✓ No tokens ever in git  
✓ Revoke anytime from HF settings  

### Colab/Kaggle (Shared Notebooks)

For shared notebooks that need a token:

1. **Create `.env` file** (it's in `.gitignore`):
   ```bash
   cp .env.example .env
   ```

2. **Add your token**:
   ```
   HF_TOKEN=hf_xxxxxxxxxxx
   ```

3. **Never commit .env** — git will ignore it automatically

✓ Scripts read from `.env` automatically  
✓ Safe for shared notebooks  

### Verify Your Setup

```bash
python scripts/setup_hf_token.py
```

---

## Quick Start

### 1️⃣ Push data to HF (one-time setup)

**Before first push**, setup your token:

```bash
# Setup (one-time)
python scripts/setup_hf_token.py
huggingface-cli login  # Paste your token

# Install
pip install huggingface_hub tqdm

# Push (automatic token detection)
python scripts/push_to_hf.py
```

✓ Reads from `/data/` directly, uploads CSVs to HF.
✓ Token is loaded automatically from `~/.huggingface/token` or `.env`.  
✓ No token parameters needed.

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
