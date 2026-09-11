# MIMIC (i2b2-format) Ground Truth Conversion — Report

**Purpose:** convert the paired `id.text` / `id.res.txt` MIMIC corpus (i2b2/n2c2-style
de-identification challenge format) into ground truth compatible with
`edgedeid.eval.metrics.compute_metrics`, so the existing evaluation pipeline
(baselines, SFT, SFT+RL, SFT+RL+QC) can be re-run unchanged against a real,
externally-annotated corpus instead of only synthetic data.

Code: [`edgedeid/eval/mimic_i2b2.py`](../../edgedeid/eval/mimic_i2b2.py),
CLI: [`scripts/convert_mimic_i2b2.py`](../../scripts/convert_mimic_i2b2.py).
Output: `evals/ext_eval/mimic_data.csv` (`Report`, `DeID`, `GT-XML` columns —
same shape as the existing `evals/ext_eval/en_data.csv`, so it drops into
`notebooks/3_eval/en_eval.ipynb`'s pattern directly, minus the
`convert_to_xml` step which is already applied).

## 1. Source format

The corpus has no separate annotation file. Instead it ships two aligned
files, each a sequence of `START_OF_RECORD=<doc>||||<line>||||…||||END_OF_RECORD`
blocks:

- `id.text` — the original note, real PHI intact.
- `id.res.txt` — the identical note with every PHI span replaced by an
  `[**Category label**]` placeholder (e.g. `[**Hospital 1**]`, `[**04-18**]`,
  `[**Last Name (un) **]`).

Record identity is the **pair** `(doc_id, line_id)` — `line_id` resets to 1
for every new `doc_id`, so it is only unique within a document (163 distinct
`doc_id`s, up to ~160 lines each, 2,434 records total). An earlier version of
the converter keyed records by `line_id` alone and silently collapsed 2,434
records down to 141; this is called out here because it's an easy trap if the
corpus format is touched again.

## 2. Method

For each aligned `(id.text, id.res.txt)` record pair:

1. **Tokenize** both bodies into whitespace / non-whitespace tokens, treating
   each `[**…**]` placeholder as one atomic token.
2. **Diff-align** the two token streams with `difflib.SequenceMatcher`. Every
   non-`equal` opcode on the `id.res.txt` side that contains a bracket token
   identifies a PHI span; the same character range on the `id.text` side is
   the PHI surrogate text (this is what a model actually needs to detect —
   it's realistic clinical text, not a placeholder).
3. **Trim** leading/trailing sentence punctuation picked up by the diff
   (e.g. a date immediately followed by `;` with no separating space) —
   otherwise ground truth like `"1992;"` would never match a model's `"1992"`
   under the containment check, silently deflating recall.
4. **Classify** the bracket's raw i2b2 label into an EdgeDeID schema tag
   (§3). Unmapped labels are kept as `None` and counted, never guessed.
5. **Re-tag** the `id.text` body with flat inline tags (`<name>`, `<date>`,
   …) for the mapped spans only, and run it through the project's existing
   `edgedeid.data.xml_utils.convert_to_xml` — the same function
   `notebooks/3_eval/en_eval.ipynb` already uses for the external English
   eval set — to produce the `<phi_entities>` ground-truth XML. This reuses
   tested infrastructure instead of hand-building the nested XML schema.

`diff_opcodes_without_bracket` (a non-`equal` opcode with no bracket at all,
i.e. `id.text` and `id.res.txt` disagreeing outside a PHI substitution) came
back **0** across the whole corpus — the two files are otherwise character-
identical, which validates the alignment approach.

## 3. Tag mapping — rule and justification

Applied in order, first match wins (checked against the label with any
trailing `" <instance-number>"` suffix stripped):

| i2b2 label pattern (case-insensitive) | EdgeDeID tag | Justification |
|---|---|---|
| `^age`, or a bare 1–2 digit number | `age` | i2b2 convention: ages < 90 are shown as the literal age number with no keyword; ages ≥ 90 are aggregated to the literal label `"Age over 90"` per HIPAA safe-harbor. |
| `date`, `month/year`, or a bare `MM-DD`/`YYYY-MM-DD`-shaped label, or a bare 4-digit year (1900–2099) | `date` | i2b2 shows partial dates as the raw digits themselves (`09-28`, `2023-05-16`) with no keyword. |
| `hospital`, `location`, `ward*name`, `clinic`, `company`, `university` | `place` | Direct 1:1 — these are all institutional/geographic entities in our `places` schema. |
| `street address`, `po box`, `address` | `address` | Direct 1:1. |
| `telephone`, `fax`, `pager` | `phone` | Combined into one bucket — our schema has a single `phones` tag with no fax/pager distinction. |
| `e-mail`/`email` | `email` | Direct 1:1. |
| `url`, `web address`, `ip address` | `url` | Direct 1:1 (none observed in this file; kept for completeness / other corpus files). |
| `device`, `serial number`, `vehicle` | `device` | Combined — none of these have their own tag in our schema. |
| `unit number`, `medical record`, `ssn`, `social security`, `health plan`, `license`, `account`, `provider number`, `job number`, `biometric` | `id` | All identifier-type PHI collapse into our single `ids` tag (which itself has a free-text `type` attribute for MRN/SSN/etc., unused here since i2b2 doesn't reliably distinguish them either). |
| `name`, `doctor`, `patient`, `initial` | `name` | Combined — i2b2 has ~30 distinct name-pattern labels (`Last Name (STitle)`, `First Name8 (NamePattern2)`, `Doctor Last Name`, `Known patient lastname`, `Initials (NamePattern4)`, …) that are all surface-matching variants of the same PHI category in our schema. `type` is inferred as `provider` if the label contains "doctor", `patient` if it contains "patient", else `other` — this is cosmetic only (see §4). |
| *(no match)* | unmapped | Counted, not guessed — see §4. |

## 4. Coverage & known limitations

Full run over all 2,434 records, 2,164 raw `[**…**]` placeholders:

| | count |
|---|---:|
| Total PHI spans recovered from diff | 2,161 |
| Mapped to an EdgeDeID tag | **2,158 (99.86%)** |
| Unmapped | 3 |
| Ambiguous multi-bracket spans (best-effort fallback) | 3 |
| Diff opcodes with no bracket (text/res mismatch outside PHI) | 0 |

Per-tag breakdown of the mapped spans:

| tag | count |
|---|---:|
| name | 1,123 |
| date | 633 |
| place | 307 |
| phone | 48 |
| age | 33 |
| address | 13 |
| id | 1 |
| email / url / device | 0 |

**What this means for the paper:**

- **MIMIC does not exercise the full EdgeDeID schema.** There are essentially
  no `emails`, `urls`, or `devices` in this corpus, and only a single `ids`
  instance (`Unit Number`) — this is a real property of de-identified
  clinical notes from this era/source, not a converter gap. Any model that
  hallucinates those tags on MIMIC input is penalized fairly (micro-F1/recall
  stay full-coverage), but `supported_entities` for per-entity/macro-F1
  reporting on this split should be restricted to
  `{names, ages, dates, places, addresses, phones, ids}` — reporting 0.0 or
  N/A for the other three would otherwise misrepresent "the model never got
  a chance to be right" as "the model failed."
- **3 unmapped spans** are all the literal i2b2 placeholder `[** **]`
  (empty — no category label at all, present verbatim in the source
  `id.res.txt` a handful of times). Nothing to map; excluded from ground
  truth.
- **3 ambiguous multi-bracket spans** (0.1% of records) occur where two or
  more PHI items appear back-to-back with no separating token the diff can
  anchor on (e.g. a name immediately followed by a phone number with only a
  dash between them). The converter falls back to tagging the whole
  combined span with the first item's category rather than guessing a split
  point — this slightly overstates that one entity's span and drops the
  second item from ground truth entirely for that record. Listed by
  `record_id` in the stats JSON (`evals/ext_eval/mimic_stats.json`) for
  manual review if wanted; too rare (3/2,434) to be worth deeper handling.
- **No entity linking / coreference.** Our schema groups repeated mentions of
  the same name under one `<name id="…">` with multiple `<value>`
  variations; i2b2's raw pairing gives no cross-reference between mentions,
  so each occurrence gets its own id. This has **no effect on the metrics**
  in `edgedeid/eval/metrics.py`, since entity matching is purely text-based
  (`extract_entities` flattens all variations into one list per tag,
  ignoring `id`/`type`) — it only affects the cosmetic shape of the
  ground-truth XML, not any score.
- **i2b2 category labels are occasionally semantically inconsistent** — e.g.
  one hospital's informal in-text reference is labeled `Last Name (un)`
  rather than `Hospital`, because it was annotated as matching a name
  surface pattern. The converter maps it to `name` per its literal label
  (faithful to the source annotation) rather than trying to override i2b2's
  own categorization — flagging this as a limitation inherited from the
  corpus, not introduced by the conversion.
- **Bare 1–2 digit numbers (age vs. 2-digit year) are heuristically
  disambiguated**, not perfectly resolved. i2b2 uses an unlabeled bare
  number both for ages < 90 and occasionally for 2-digit year shorthand
  (`MI '94`). The converter treats all bare 1–2 digit brackets as `age`
  (the standard i2b2 convention, since ages ≥ 90 get the distinct
  `"Age over 90"` label) — a small, undetermined fraction of these may
  actually be shorthand years mislabeled as ages. Not correctable from the
  bracket alone; worth a one-line caveat in the paper's limitations section.

## 5. How to use

```python
import pandas as pd
df = pd.read_csv("evals/ext_eval/mimic_data.csv")
# df["Report"]  -> raw MIMIC note, feed to model as input
# df["GT-XML"]  -> ground truth <phi_entities> XML, ready for compute_metrics

from edgedeid.eval.metrics import compute_metrics
result = compute_metrics(
    predictions=model_outputs,          # your model's completions on df["Report"]
    ground_truths=df["GT-XML"].tolist(),
    source_notes=df["Report"].tolist(),
    supported_entities={"name", "age", "date", "place", "address", "phone", "id"},
)
```

No changes to `edgedeid/eval/metrics.py` are needed — this only produces a
new `(predictions, ground_truths, source_notes)` triple in the same shape
the synthetic-data eval already consumes.
