# EdgeDeID HuggingFace Collection

This document catalogs the models and datasets available in the EdgeDeID HuggingFace collection for de-identification tasks.

---

## 📦 Models

### 1. Qwen3-1.7B-Base_SFT
- **Repository**: https://huggingface.co/EdgeDeID/Qwen3-1.7B-Base_SFT
- **Type**: Language Model (Supervised Fine-Tuned)
- **Base Model**: Qwen 3 (1.7B parameters)
- **Purpose**: De-identification task with SFT training on synthetic dataset
- **Access**: [Model Card](https://huggingface.co/EdgeDeID/Qwen3-1.7B-Base_SFT)


**Use Case:**
- Fine-tuned version of Qwen 3 for de-identifying medical/clinical text
- Smaller 1.7B parameter model for efficient inference
- Trained on EdgeDeID synthetic de-identification dataset

---

### 2. Qwen3-0.6B-Base_SFT
- **Repository**: https://huggingface.co/EdgeDeID/Qwen3-0.6B-Base_SFT
- **Type**: Language Model (Supervised Fine-Tuned)
- **Base Model**: Qwen 3 (0.6B parameters)
- **Purpose**: De-identification task with SFT training on synthetic dataset
- **Access**: [Model Card](https://huggingface.co/EdgeDeID/Qwen3-0.6B-Base_SFT)

**Use Case:**
- Lightweight 0.6B parameter model for resource-constrained environments
- Fine-tuned for de-identification tasks
- Ideal for edge deployment scenarios
- Trained on EdgeDeID synthetic de-identification dataset

---

## 📊 Datasets

### 1. Synthetic Dataset
- **Repository**: https://huggingface.co/datasets/EdgeDeID/synthetic-dataset
- **Type**: Dataset (HuggingFace Datasets)
- **Purpose**: Training data for de-identification models
- **Access**: [Dataset Card](https://huggingface.co/datasets/EdgeDeID/synthetic-dataset)

**Expected Structure:**
- Training split with labeled examples
- Evaluation/test split for benchmarking
- Feature columns for text and de-identification labels
- Metadata documenting generation process and data characteristics

---

## 🔍 Verification & Content Access

### Prerequisites
- Python 3.11+ 
- `huggingface-hub` library (v0.35.0+)
- Internet connection for HuggingFace Hub access

### Installation
```bash
pip install huggingface-hub transformers datasets
```

### Verify Models & Dataset Content

#### 1. List Model Files
```python
from huggingface_hub import list_repo_files

# Check Qwen3-1.7B-Base_SFT model
files_1_7b = list_repo_files("EdgeDeID/Qwen3-1.7B-Base_SFT", repo_type="model")
print("Qwen3-1.7B-Base_SFT files:")
for f in files_1_7b:
    print(f"  - {f}")

# Check Qwen3-0.6B-Base_SFT model
files_0_6b = list_repo_files("EdgeDeID/Qwen3-0.6B-Base_SFT", repo_type="model")
print("\nQwen3-0.6B-Base_SFT files:")
for f in files_0_6b:
    print(f"  - {f}")
```

#### 2. Load Model Information
```python
from huggingface_hub import model_info, dataset_info

# Get model metadata
model_1_7b = model_info("EdgeDeID/Qwen3-1.7B-Base_SFT")
print(f"Model: {model_1_7b.id}")
print(f"Downloads: {model_1_7b.downloads}")
print(f"Likes: {model_1_7b.likes}")
print(f"Tags: {model_1_7b.tags}")

model_0_6b = model_info("EdgeDeID/Qwen3-0.6B-Base_SFT")
print(f"\nModel: {model_0_6b.id}")
print(f"Downloads: {model_0_6b.downloads}")
print(f"Likes: {model_0_6b.likes}")

# Get dataset metadata
dataset = dataset_info("EdgeDeID/synthetic-dataset")
print(f"\nDataset: {dataset.id}")
print(f"Downloads: {dataset.downloads}")
print(f"Likes: {dataset.likes}")
```

#### 3. Load & Inspect Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load 1.7B model
tokenizer = AutoTokenizer.from_pretrained("EdgeDeID/Qwen3-1.7B-Base_SFT")
model = AutoModelForCausalLM.from_pretrained("EdgeDeID/Qwen3-1.7B-Base_SFT")

print(f"Model type: {type(model)}")
print(f"Model size: {model.num_parameters():,} parameters")
print(f"Tokenizer vocab size: {len(tokenizer)}")
```

#### 4. Load & Inspect Dataset
```python
from datasets import load_dataset

# Load synthetic dataset
dataset = load_dataset("EdgeDeID/synthetic-dataset")
print(f"Dataset splits: {dataset.keys()}")
print(f"Dataset info: {dataset}")

# Check first example
print("\nFirst example:")
print(dataset['train'][0])
```

#### 5. Quick Verification Script
```bash
python fetch_hf_info.py
```
This will generate `hf_collection_info.json` with detailed metadata.

---

## 📝 Notes

- **Authentication**: If repositories are private, ensure you have HuggingFace credentials configured
  ```bash
  huggingface-cli login
  ```
- **Model Access**: Both models are based on Qwen 3, which is open-source but ensure compliance with its license
- **Dataset License**: Check the dataset card for licensing information
- **Memory Requirements**:
  - 1.7B model: ~3.5-4GB GPU VRAM (FP16) or 7GB (FP32)
  - 0.6B model: ~1.5-2GB GPU VRAM (FP16) or 3-4GB (FP32)

---

## 🚀 Quick Start

To use these models for de-identification:

```python
from transformers import pipeline

# Load the de-identification pipeline
de_id = pipeline("text-generation", 
                 model="EdgeDeID/Qwen3-1.7B-Base_SFT")

# De-identify text
text = "John Doe was born on January 15, 1990, and lives at 123 Main St, New York."
result = de_id(text, max_length=100)
print(result)
```

---

**Last Updated**: 2026-05-06  
**Collection Owner**: EdgeDeID  
**Documentation Version**: 1.0