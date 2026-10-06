# PhysioNet deid gold standard (MIMIC nursing notes)

Downloaded 2026-09-30 from https://physionet.org/content/deid/1.1/ (Open Access,
Open Data Commons Attribution License v1.0). Cite the PhysioNet deid project and
Neamatullah et al. (2008) when using these files.

- `id-phi.phrase`: HUMAN gold standard, one PHI span per line
  (`patient note start end type phrase`; 1,779 spans over the 2,434 notes in `../../../id.text`).
  Offsets are one character short of `edgedeid.eval.mimic_i2b2.parse_records` bodies.
- `id.deid`: the same gold spans as offsets only.
- `GSoutput_id.phi`: PHI detected by the `deid.pl` tool itself (not a gold standard);
  `../../../id.res.txt` is that tool's text output, which earlier evaluations mistook for gold.
- `shift.txt`: per-patient date shifts. `README.txt`: PhysioNet's file description.

The GPL-licensed deid software files (`runStat.pl`, `stat.pm`, `COPYING`) are not tracked.
Converted ground truth: `../mimic_data_gold.csv` via `scripts/convert_mimic_i2b2.py --gold`.
