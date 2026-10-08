"""CLI: convert the paired id.text/id.res.txt MIMIC corpus into an
eval-ready CSV (Report, DeID, GT-XML — same shape as evals/ext_eval/en_data.csv)
plus a JSON stats summary used to write the conversion report.

Usage:
    py -3 scripts/convert_mimic_i2b2.py --text id.text --res id.res.txt \
        --out evals/ext_eval/mimic_data.csv --stats-out evals/ext_eval/mimic_stats.json

    # HUMAN gold standard (PhysioNet deid 1.1 id-phi.phrase). id.res.txt is the
    # output of the deid.pl tool, not an annotation -- prefer this for scoring.
    py -3 scripts/convert_mimic_i2b2.py --text id.text \
        --gold evals/ext_eval/physionet_deid/id-phi.phrase \
        --out evals/ext_eval/mimic_data_gold.csv --stats-out evals/ext_eval/mimic_gold_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from edgedeid.eval.mimic_i2b2 import convert_corpus, convert_corpus_gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="id.text")
    ap.add_argument("--res", default="id.res.txt")
    ap.add_argument("--out", default="evals/ext_eval/mimic_data.csv")
    ap.add_argument("--stats-out", default="evals/ext_eval/mimic_stats.json")
    ap.add_argument("--gold", default=None,
                    help="PhysioNet id-phi.phrase: build ground truth from the human "
                         "gold standard instead of diffing against --res")
    args = ap.parse_args()

    if args.gold:
        records = convert_corpus_gold(args.text, args.gold)
        by_tag = Counter(s.tag for r in records for s in r.spans)
        by_type = Counter(s.label for r in records for s in r.spans)
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Report", "DeID", "GT-XML"])
            for r in records:
                w.writerow([r.source_note, r.tagged_note, r.ground_truth_xml])
        stats = {
            "source": "human gold standard (PhysioNet deid 1.1, id-phi.phrase)",
            "n_records": len(records),
            "records_with_phi": sum(bool(r.spans) for r in records),
            "gold_spans": sum(by_tag.values()),
            "by_schema_tag": dict(by_tag.most_common()),
            "by_gold_type": dict(by_type.most_common()),
        }
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2))
        return

    records = convert_corpus(args.text, args.res)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Report", "DeID", "GT-XML"])
        for r in records:
            w.writerow([r.source_note, r.tagged_note, r.ground_truth_xml])

    mapped_by_tag: Counter[str] = Counter()
    unmapped_labels: Counter[str] = Counter()
    total_spans = 0
    total_ambiguous = 0
    total_no_bracket = 0

    for r in records:
        total_ambiguous += r.ambiguous_multi_bracket
        total_no_bracket += r.diff_without_bracket
        for span in r.spans:
            total_spans += 1
            if span.tag is not None:
                mapped_by_tag[span.tag] += 1
            else:
                label = span.label
                import re as _re
                label = _re.sub(r"\s+\d+$", "", label).strip()
                unmapped_labels[label] += 1

    total_mapped = sum(mapped_by_tag.values())
    total_unmapped = sum(unmapped_labels.values())

    stats = {
        "n_records": len(records),
        "total_phi_spans": total_spans,
        "mapped_spans": total_mapped,
        "unmapped_spans": total_unmapped,
        "mapped_by_schema_tag": dict(mapped_by_tag.most_common()),
        "unmapped_by_raw_label": dict(unmapped_labels.most_common()),
        "ambiguous_multi_bracket_spans": total_ambiguous,
        "diff_opcodes_without_bracket": total_no_bracket,
    }
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
