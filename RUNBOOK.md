# EdgeDeID — Final Full-Run Runbook

Everything needed to regenerate every number in the paper, in dependency order,
for one clean run on a GPU machine. Written for Bardia.

**Target:** ARR October cycle — submission **12 Oct 2026**; results needed by
**8 Oct**.

What changed since the last run, in one paragraph: the ablation ladder is now
A–E (§3); there is a **new QC verifier** (§5) that replaces the self-assessment
tag that never appeared; the latency benchmark was **broken** and is replaced
by a script (§6); the definition of "complete" is now the zero-leak rule
everywhere (§7). Everything new is smoke-tested end-to-end on CPU against the
real TRL/PEFT APIs — see §9 for what was and was not tested.

---

## 0. Setup

```bash
git pull && pip install -e .          # pyproject now also pulls peft + psutil
echo "HF_TOKEN=hf_..." > .env         # scripts read .env; checkpoints are private
python -m spacy download en_core_web_sm    # optional: QC Tier-3 only
```

Versions: `pyproject.toml` pins `transformers<5`, `trl>=0.23`. The new scripts
were developed on transformers 5.8 / trl 1.4 and are written to tolerate both:
TRL configs are built by filtering keyword arguments against the installed
version, and anything ignored is printed (e.g. `max_prompt_length` exists only
in trl<1.0 — where its 512 default would left-truncate our prompts, so the
scripts set it to `None`). **If a script prints "not supported by this TRL
version, ignored: [...]", read the list.**

---

## 1. What produces what

| Paper element | Produced by | Hardware | Time (1 GPU) |
|---|---|---|---|
| Main comparison, zero-leak, per-entity | `notebooks/3_eval/baselines.ipynb` §1–7 | GPU | ~30 min |
| Ablation ladder A–E + decoding ablation | `notebooks/3_eval/ablation.ipynb` | GPU | ~20 min |
| QC tier-by-tier table | `notebooks/3_eval/qc_ablation.ipynb` | GPU | ~10 min |
| Significance (paired bootstrap) | `scripts/compare_runs.py` | CPU | <1 min |
| MIMIC generalization | `scripts/score_mimic.py` | CPU | <1 min |
| Robustness / noise | `scripts/robustness_eval.py` | GPU | ~15 min |
| **QC verifier (new)** | `scripts/build_qc_dataset.py` → `train_qc_sft.py` → `train_qc_grpo.py` → `eval_qc_verifier.py` | GPU | ~1.5–3 h |
| **Latency (fixed)** | `scripts/benchmark_latency.py` | CPU + GPU | ~1–2 h (CPU) |

**Do not use `baselines.ipynb` §8 (latency) any more** — see §6.

---

## 2. Evaluation sets

| Set | File | Size | Role |
|---|---|---|---|
| Radiology benchmark | `evals/ext_eval/revizied_evals.xlsx` | 100 notes | primary |
| MIMIC (i2b2 format) | `merged_all_models.csv` (13 systems' outputs) | 2,434 records | external generalization |
| Synthetic train split | `EdgeDeID/synthetic-dataset` `train.csv` | 25,972 notes | **only** source of QC-verifier training data |

Neither test set is ever used for training or threshold selection. MIMIC is
PHI-sparse (60.3% of records have no PHI), so zero-leak is not comparable there
and is omitted from the MIMIC table.

---

## 3. Ablation ladder — now continuous A–E

| Row | Config | How |
|---|---|---|
| A | Base, 0-shot | inference (`Qwen/Qwen3-0.6B-Base`) |
| B | SFT (pseudo-label) | inference (`EdgeDeID/Qwen3-0.6B-Base_SFT`) |
| C | SFT + QC | derived from B |
| D | SFT + GRPO | inference (`EdgeDeID/Qwen3-0.6B-GRPO`) |
| E | SFT + GRPO + QC | derived from D |

> ⚠️ **`rm -f ablation/*_predictions.csv` before running.** Under the old
> letters `C_predictions.csv` was SFT; now `C` is SFT+QC. A stale cache
> silently yields wrong numbers.

---

## 4. Run order

Run top to bottom. Steps marked **CPU** can run anywhere, in parallel.

### 4.1 Main comparison — GPU
```bash
jupyter nbconvert --to notebook --execute notebooks/3_eval/baselines.ipynb
```
Skip/ignore §8 (latency) — replaced by §6 below. Writes
`artifacts/baselines/` including `predictions/pred_<System>.csv`.

### 4.2 Ablation + decoding — GPU
```bash
rm -f ablation/*_predictions.csv
jupyter nbconvert --to notebook --execute notebooks/3_eval/ablation.ipynb
```

### 4.3 QC tier-by-tier — GPU
```bash
jupyter nbconvert --to notebook --execute notebooks/3_eval/qc_ablation.ipynb
```
**Set `FORCE_REGEN = True`** in §1 of the notebook. Its calibration table uses
the self-assessment tag of the old GRPO model; we expect 0/100 tag coverage
again, which *confirms* the finding (§8). The tier table (Raw / +Repair /
+Tier-2 / +Tier-3) is the part that goes in the paper.

### 4.4 Significance — CPU (needs 4.2)
```bash
python scripts/compare_runs.py --rows B D     # SFT vs SFT+GRPO
python scripts/compare_runs.py --rows D E     # GRPO vs GRPO+QC
```

### 4.5 MIMIC — CPU
```bash
python scripts/score_mimic.py --predictions merged_all_models.csv --out-dir artifacts/mimic
```
Quote the **coverage-adjusted** table.

### 4.6 Robustness — GPU
```bash
python scripts/robustness_eval.py --model EdgeDeID/Qwen3-0.6B-Base_SFT --all
```

### 4.7 QC verifier — GPU (see §5)
### 4.8 Latency — CPU and GPU (see §6)

### 4.9 Audit — CPU, last
```bash
python scripts/audit_paper_tables.py --artifacts artifacts
```
Checks every table in `draft.tex` against its CSV and that every F1 is the
harmonic mean of the P and R beside it. Non-zero exit on a real mismatch.

---

## 5. The QC verifier (new)

### Why
The self-assessment tag was trained with GRPO and never appeared. Cause: SFT
targets end at `</phi_entities>`, so the policy put ~0 probability on anything
after it; every sampled completion scored 0 on the tag reward, the term had
zero within-group variance, and GRPO received no gradient from it. **GRPO can
sharpen a behaviour the policy already produces; it cannot create one.**

The verifier fixes this by construction. The candidate annotation goes in the
**prompt**; the model writes only the verdict:

```
<instructions>…</instructions>  <note>…</note>
<phi_entities>…candidate…</phi_entities>
<qc_status>                      ← prompt ends here
complete</qc_status>  |  incomplete</qc_status>    ← all the model writes
```

* Loss/reward touch only the verdict → a corrupted candidate can never teach
  the model to omit entities.
* Two possible outputs → both in-support from step one → GRPO groups have
  variance immediately.
* The candidate can come from **any** system → a system-agnostic review filter
  ("assume someone else de-identified this").
* Trained as a **LoRA adapter**: extraction runs with it OFF (identical
  weights, identical numbers), verification with it ON.

### Steps
```bash
# 1. data — CPU, ~2 min, from the synthetic TRAIN split only
python scripts/build_qc_dataset.py --out-dir data/qc

#    optional, adds real baseline mistakes on training notes (~15 min GPU):
python scripts/generate_qc_candidates.py --out artifacts/qc_candidates/train_candidates.csv
python scripts/build_qc_dataset.py --out-dir data/qc \
    --extra-candidates artifacts/qc_candidates/train_candidates.csv

# 2. stage 1 — SFT LoRA adapter (~30–60 min)
python scripts/train_qc_sft.py --data-dir data/qc --out-dir checkpoints/qc_sft

# 3. stage 2 — GRPO on the verdict (~30–60 min)
python scripts/train_qc_grpo.py --data-dir data/qc --sft-adapter checkpoints/qc_sft \
    --out-dir checkpoints/qc_grpo --max-steps 300

# 4. evaluate as a review filter on BOTH test sets
python scripts/make_wide_predictions.py --notes-from ablation/B_predictions.csv \
    --pred-dir artifacts/baselines/predictions --out artifacts/qc_eval/radiology_wide.csv
python scripts/eval_qc_verifier.py \
    --config zero_shot= --config sft=checkpoints/qc_sft --config grpo=checkpoints/qc_grpo \
    --testset radiology=artifacts/qc_eval/radiology_wide.csv \
    --testset mimic=merged_all_models.csv \
    --val data/qc/qc_sft_val.jsonl --out-dir artifacts/qc_eval
#    add --max-per-system 400 if MIMIC scoring (31k pairs x 3 configs) is too slow
```

### What to watch
* **Stage 1 log**: `val_auroc` before vs after. Before ≈ 0.5 (the extraction
  model has never seen the task); after should be well above.
* **Stage 2 log**: `frac_reward_zero_std` must be **< 1** — it is the share of
  groups that gave no gradient. If it stays ≈ 1, raise `--temperature`. Also
  `reward` should rise.
* **Memory**: SFT computes full-vocabulary logits for logging; batch 4 × ~2.5k
  tokens needs ~15 GB transient. On a 24 GB card use `--batch-size 2
  --grad-accum 8`.

### Design details worth knowing
* **Labels** come from `edgedeid.eval.metrics.note_is_complete` — the exact
  zero-leak rule (verified identical on 32,142 prediction/reference pairs).
  Every candidate is re-labelled by this rule, never assumed from the
  perturbation: dropping the variation "Chen" while "Wei Chen" survives is
  still complete.
* **Training data**: gold untouched; harmless edits; spurious extra entities
  (over-redaction ≠ incomplete); single/multiple drops; partial spans ("Amit
  Kumar" → "Amit"); whole-category drops (a system blind to places); drop +
  add. 30% incomplete for SFT, 45% for GRPO — deliberately not tiny: if
  P(incomplete) collapses to ~3% after SFT, a group of 8 contains an
  "incomplete" sample only ~22% of the time and GRPO starves again.
* **Shortcut audit**: every candidate is serialized canonically (byte-exact to
  the gold format on 99.3% of training notes), ids are renumbered, cosmetic
  edits are applied uniformly. Note-blind features (entity count, empty
  categories, length, missing attributes) predict the label at AUROC
  0.47–0.57 — the verifier has to read the note.
* **Reward**: well-formed +0.2, then correct +0.8 / over-cautious +0.4 /
  overconfident 0. Correct 1.0 > over-cautious 0.6 > overconfident 0.2 >
  malformed 0. The 0.5 over-caution value puts the reward-optimal threshold at
  P(incomplete) > 1/3.
* **KL**: `--beta 0` by default. With a LoRA policy, TRL's reference model is
  the adapter-disabled model — the plain extraction model, which cannot
  verify — so a KL term would pull the verifier back toward chance.
* **Scoring** needs no generation: the two verdicts differ at their first
  token (`in` vs `complete`), so one forward pass gives P(incomplete).
* **Thresholds** are picked on the synthetic *validation* split and applied
  unchanged to the test sets (1/3, 0.5, and "catch ≥95% on validation").

### Outputs (`artifacts/qc_eval/`)
* `summary.csv` — AUROC, AUPRC, ECE, Brier, and at each threshold: flag rate
  (review budget), catch rate, residual leak rate (leaky notes released
  without review); for all systems pooled, `self` (EdgeDeID-SFT), `others`,
  and each system. **zero_shot → sft → grpo is the ablation; sft → grpo is the
  RL contribution.**
* `curves.csv` — review-budget curves: the headline figure.
* `scores.csv` — per pair, for bootstrap CIs later.

---

## 6. Latency — the old benchmark was broken

### Diagnosis
| # | Problem | Evidence |
|---|---|---|
| 1 | **No common device.** `CUDA_VISIBLE_DEVICES=""` was set mid-kernel after §4 had initialized CUDA, so it did nothing. Only our model (it exposes `.model`) was force-moved to CPU fp32. Pipeline baselines default to `cuda:0` (transformers 5.x: `device=None → 0`), OPF picks CUDA explicitly, GLiNER/Presidio/scispaCy stay on CPU. | Pipeline baselines' "CPU" latency equals their batched-GPU §4 timing (RoBERTa 12.3 vs 12.9 ms). Re-measured on a real CPU: **BERT-i2b2 = 286 ms/note**, not 9 ms. |
| 2 | **"Peak RSS" was a delta** of current RSS inside one long-lived process; freed memory is reused. | Values of 0 and **−1 MB** in the table. |
| 3 | GGUF cell spawned `llama-cli` **per note** → timed model loading. | (never run) |
| 4 | Checkpoints saved with `use_cache: false` (from gradient-checkpointed SFT). | On transformers 5.8 `generate()` ignores it (tested: cache stays on); 4.x may not. Now pinned `use_cache=True` in every `generate()` call. |

The ~23 s/note for our model is therefore most likely a **real CPU-fp32
number**, compared against GPU baselines. It is also inherently larger than an
encoder's: the model writes ~180 output tokens one at a time (median prompt
1,060 tokens, of which the fixed instruction block is 867 = 82%).

### Fixed protocol
```bash
python scripts/benchmark_latency.py --device cpu  --notes-from ablation/B_predictions.csv
python scripts/benchmark_latency.py --device cuda --notes-from ablation/B_predictions.csv
```
Each model runs in a **fresh subprocess** with the device fixed before Python
starts; the device it actually ran on is read back and reported per row
(`device_verified`, flagged if wrong). Peak RSS is the true process
high-water mark. Our models additionally get prompt/output token counts,
prefill time, and decode tokens/s. Hardware goes into `_meta.json`.

`--n 20` default. The 0-shot base model never stops (1,024 tokens every note)
and takes ~2 min/note on CPU — drop it with `--models` if short on time.

### Edge number (optional, recommended if the paper keeps "edge")
The real edge runtime is a quantized GGUF under llama.cpp:
```bash
python llama.cpp/convert_hf_to_gguf.py <local SFT checkpoint> --outfile sft-f16.gguf
llama.cpp/build/bin/llama-quantize sft-f16.gguf sft-Q4_K_M.gguf Q4_K_M
llama.cpp/build/bin/llama-server -m sft-Q4_K_M.gguf -c 8192 -t 8 --port 8080 &
python scripts/benchmark_latency.py --device cpu --models GGUF \
    --llama-server http://127.0.0.1:8080 --gguf-label EdgeDeID-SFT-Q4_K_M
```
Add `--cache-prompt` for a second row that reuses the shared 867-token prefix.
Accuracy of the quantized model is **not** measured by this — check it before
quoting it.

---

## 7. Definition of "complete" — now one rule everywhere

`note_recall` / `note_is_complete` (`edgedeid/eval/metrics.py`): a reference
entity counts as caught only if a prediction **of the same category fully
contains it** — exactly the zero-leak rule. `qc/calibration.py` and the new
verifier use it. Previously calibration used `rl.rewards.entity_recall`
(bidirectional, category-agnostic), under which a partial span "Amit" for
"Amit Kumar" counted as caught although zero-leak counts it a leak.
`QC_COMPLETE_THRESHOLD` stays 1.0 (the paper text still says 0.7 — to fix).

---

## 8. Findings the re-run should reproduce

- SFT is the best extraction configuration; neither GRPO nor QC improves it.
- GRPO vs SFT: F1 −0.0051 [−0.0089, −0.0020], recall −0.0103
  [−0.0167, −0.0051], zero-leak −0.0800 [−0.1300, −0.0300]; precision
  +0.0004 not significant.
- QC Tier-3: recall +0.0017 and zero-leak +0.0100, neither significant;
  precision −0.0536, F1 −0.0257, hallucination +0.0536, all significant.
- MIMIC: SFT 0.984 → 0.423 F1, beaten by RoBERTa-i2b2 (0.571); a precision
  collapse on ICU shorthand. Annotation-convention mismatch explains at most
  ~20% of the false positives (removing ages entirely: 0.423 → 0.439); the
  rest are register errors — drug names and abbreviations tagged as names
  (VANCOMYCIN, FOLEY, A+Ox3), military times read as years (1930 = 7:30 pm).
- Self-assessment tag coverage: 0/100 (expected again with `FORCE_REGEN`).

---

## 9. What was tested, and what was not

Tested locally (CPU, transformers 5.8, trl 1.4, peft 0.18):
- The full verifier chain end-to-end on a tiny random model: build → SFT →
  GRPO → assemble → evaluate. Wiring only; says nothing about accuracy.
- Dataset builder on the real 25,972-note train split.
- Latency script on real models on CPU.

**Not tested:** anything on GPU; any real training run; transformers 4.x /
trl 0.23. First real signal will be the stage-1 `val_auroc` — if it is ≈0.5
after training, stop and check before running stage 2.

---

## 10. What to send back

`artifacts/` (all subfolders), `ablation/`, `checkpoints/qc_sft/` and
`checkpoints/qc_grpo/` (adapters only, a few MB each), the training logs of
§5 steps 2–3, and `artifacts/latency/*_meta.json` (hardware for the paper).
