# Hugging Face Dataset Pipeline

Comprehensive guide to push/pull EdgeDeID synthetic data to/from Hugging Face Hub.

**Dataset repo:** `https://huggingface.co/datasets/EdgeDeID/synthetic-dataset`

## Dataset Structure

```
synthetic-dataset/  (HF repo root)
├── deepseek/
│   ├── output_1_deepseek.csv
│   ├── output_2_deepseek.csv
│   ├── ... (output_1 through output_6)
│   └── deepseek_pseudolabeled.csv
├── haiku/
│   ├── output_1_haiku.csv
│   ├── output_2_haiku.csv
│   ├── ... (output_1 through output_7)
│   └── haiku_pseudolabeled.csv
└── hermes/
    ├── output_1_hermes.csv
    ├── output_2_hermes.csv
    ├── ... (output_1 through output_17)
    └── hermes_pseudolabeled.csv
```

**Total:** 30 raw outputs + 3 pseudolabeled = 33 CSV files (~500 MB–1 GB)

---

## Part 1: Push Data to HF (One-time setup)

### 1. Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login  # Paste your HF token
git lfs install        # Enable Git LFS for large files
```

### 2. Clone HF dataset repo locally

```bash
git clone https://huggingface.co/datasets/EdgeDeID/synthetic-dataset
cd synthetic-dataset
```

### 3. Copy data from EdgeDeID repo

From your EdgeDeID directory, copy the data. On Windows (PowerShell):

```powershell
# Navigate to EdgeDeID root, then copy directories
Copy-Item -Path "data/deepseek" -Destination "C:\path\to\synthetic-dataset\" -Recurse -Force
Copy-Item -Path "data/haiku"    -Destination "C:\path\to\synthetic-dataset\" -Recurse -Force
Copy-Item -Path "data/hermes"   -Destination "C:\path\to\synthetic-dataset\" -Recurse -Force
```

Or on Linux/Mac:

```bash
cp -r data/deepseek /path/to/synthetic-dataset/
cp -r data/haiku    /path/to/synthetic-dataset/
cp -r data/hermes   /path/to/synthetic-dataset/
```

### 4. Configure Git LFS for CSVs

In the `synthetic-dataset` repo, set up LFS tracking:

```bash
cd synthetic-dataset
git lfs track "*.csv"
```

This creates `.gitattributes`. Add it to git:

```bash
git add .gitattributes
```

### 5. Add and push all data

```bash
git add .
git commit -m "Add all synthetic data (deepseek, haiku, hermes) - raw + pseudolabeled"
git push
```

**First push may take 10–30 minutes** (LFS uploads). Monitor progress in the terminal.

### 6. Verify on HF Hub

Visit `https://huggingface.co/datasets/EdgeDeID/synthetic-dataset` and confirm all files appear.

---

## Part 2: Pull Data in Colab/Kaggle (Programmatic)

### Option A: Python Script (Recommended)

**Installation in Colab/Kaggle cell:**

```python
!pip install huggingface_hub

from huggingface_hub import snapshot_download
import os

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="EdgeDeID/synthetic-dataset",
    repo_type="dataset",
    local_dir="./data",           # Downloads to ./data/
    local_dir_use_symlinks=False  # Avoid symlink issues on Colab
)

print(f"Dataset downloaded to: {dataset_path}")

# Verify structure
import os
for root, dirs, files in os.walk("data"):
    level = root.replace("data", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files per folder
        print(f"{subindent}{file}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")
```

**Output:** Creates directory structure:
```
./data/
├── deepseek/
│   ├── output_1_deepseek.csv
│   ├── ...
│   └── deepseek_pseudolabeled.csv
├── haiku/
├── hermes/
```

### Option B: Selective Download (Download only pseudolabeled)

If bandwidth is limited, download only the cleaned pseudolabeled CSVs:

```python
from huggingface_hub import hf_hub_download
import os

# Create directory
os.makedirs("data", exist_ok=True)

teachers = ["deepseek", "haiku", "hermes"]
for teacher in teachers:
    file_path = hf_hub_download(
        repo_id="EdgeDeID/synthetic-dataset",
        filename=f"{teacher}/{teacher}_pseudolabeled.csv",
        repo_type="dataset",
        local_dir="./data"
    )
    print(f"Downloaded: {file_path}")
```

### Option C: CLI (Quick, one-liner)

For Colab, in a shell cell:

```bash
!pip install huggingface_hub
!huggingface-cli download EdgeDeID/synthetic-dataset --repo-type dataset --local-dir ./data
```

---

## Part 3: Integration with Data Pipeline Notebooks

### Example: `data_prep/combine.ipynb`

At the top of the notebook, add a cell to download data:

```python
# Cell 0: Download data from HF (if running on Colab/Kaggle)

import os
from huggingface_hub import snapshot_download

# Check if data already exists
if not os.path.exists("data/deepseek/deepseek_pseudolabeled.csv"):
    print("Downloading EdgeDeID synthetic data from Hugging Face...")
    snapshot_download(
        repo_id="EdgeDeID/synthetic-dataset",
        repo_type="dataset",
        local_dir="./data",
        local_dir_use_symlinks=False
    )
    print("✓ Data downloaded successfully!")
else:
    print("✓ Data already exists locally")
```

Then, your existing `combine.ipynb` code works as-is (reads from `data/deepseek/`, etc.).

### Example: `data_prep/pseudolabel.ipynb`

Similarly, at the start:

```python
# Download raw outputs if needed
from huggingface_hub import hf_hub_download
import os

dataset_name = "hermes"  # or "haiku", "deepseek"

# Download all raw outputs for this teacher
os.makedirs(f"data/{dataset_name}", exist_ok=True)
for i in range(1, 20):  # Adjust range per teacher
    try:
        hf_hub_download(
            repo_id="EdgeDeID/synthetic-dataset",
            filename=f"{dataset_name}/output_{i}_{dataset_name}.csv",
            repo_type="dataset",
            local_dir="./data"
        )
    except Exception:
        break  # Stop when output_N doesn't exist
```

---

## Part 4: File Sizes & Download Times

| Teacher   | Raw Outputs | Pseudolabeled | Total   |
|-----------|-------------|---------------|---------|
| deepseek  | ~150 MB     | ~20 MB        | ~170 MB |
| haiku     | ~180 MB     | ~20 MB        | ~200 MB |
| hermes    | ~280 MB     | ~25 MB        | ~305 MB |
| **Total** | ~610 MB     | ~65 MB        | **~675 MB** |

**Download times (typical):**
- Full dataset: 2–5 min (broadband)
- Pseudolabeled only: 30–60 sec

---

## Part 5: Troubleshooting

### "Token not found" during `huggingface-cli login`

Generate a token on HF Hub:
1. Go to `https://huggingface.co/settings/tokens`
2. Click "New token" (type: **Read**)
3. Copy and paste into terminal prompt

### "No space left on device" (Colab)

Colab has ~15 GB storage. The full dataset (~675 MB) fits. If low on space:

```python
# Download pseudolabeled only (much smaller)
teachers = ["deepseek", "haiku", "hermes"]
for teacher in teachers:
    hf_hub_download(
        repo_id="EdgeDeID/synthetic-dataset",
        filename=f"{teacher}/{teacher}_pseudolabeled.csv",
        repo_type="dataset",
        local_dir="./data"
    )
```

### Dataset not showing up on HF Hub

After push:
1. Refresh the page (`Ctrl+Shift+R`)
2. Wait 2–5 minutes (HF indexing)
3. Check repo settings: ensure it's not **private** (unless intentional)

---

## Part 6: Updating Data (Adding new runs)

When you generate new outputs (e.g., `output_8_haiku.csv`):

```bash
cd synthetic-dataset
# Copy new file
cp ../EdgeDeID/data/haiku/output_8_haiku.csv ./haiku/

# Commit and push
git add .
git commit -m "Add output_8_haiku.csv"
git push
```

Colab users automatically pull the latest on next `snapshot_download()` call (no cache by default).

---

## Part 7: Quick Copy-Paste Commands

**Push (local machine, one-time):**
```bash
huggingface-cli login
git clone https://huggingface.co/datasets/EdgeDeID/synthetic-dataset
cp -r data/* synthetic-dataset/
cd synthetic-dataset
git lfs track "*.csv"
git add .
git commit -m "Add synthetic data"
git push
```

**Pull (Colab, in notebook cell):**
```python
!pip install huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download("EdgeDeID/synthetic-dataset", repo_type="dataset", local_dir="./data")
```

---

## References

- [HF Hub Datasets Documentation](https://huggingface.co/docs/hub/datasets)
- [Git LFS Setup](https://git-lfs.com/)
- [`huggingface_hub` Python API](https://huggingface.co/docs/hub/security-tokens)
