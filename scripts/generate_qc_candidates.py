#!/usr/bin/env python3
"""OPTIONAL: real system outputs on TRAINING notes, as extra verifier candidates.

Synthetic perturbations of the gold (``build_qc_dataset.py``) teach the
verifier what incompleteness looks like; real systems' mistakes show what it
looks like *in practice* — e.g. Presidio blind to places, BERT-i2b2 missing
URLs. This runs a few fast baselines over the same training-note pools the
builder uses and writes ``original, candidate, source`` rows for its
``--extra-candidates`` flag.

Only TRAINING notes are used. ``--n-notes / --split / --seed`` must match the
values passed to build_qc_dataset.py so these notes fall in the same pools;
candidates on notes outside a pool are dropped by the builder anyway.

Our own SFT model is not in the default list: it was trained on these notes,
so its outputs here are almost always complete and add little signal.

Usage
-----
    python scripts/generate_qc_candidates.py --out artifacts/qc_candidates/train_candidates.csv
    python scripts/build_qc_dataset.py --out-dir data/qc \
        --extra-candidates artifacts/qc_candidates/train_candidates.csv
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from edgedeid.qc.train_utils import load_env  # noqa: E402
from edgedeid.qc.verifier_data import split_notes  # noqa: E402
from scripts.benchmark_latency import REGISTRY, _build, _place  # noqa: E402

DEFAULT_SYSTEMS = "Presidio,BERT-i2b2,RoBERTa-i2b2,GLiNER"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS)
    ap.add_argument("--notes-per-system", type=int, default=1500,
                    help="training notes sampled from the pools for each system")
    ap.add_argument("--n-notes", type=int, default=8000, help="must match build_qc_dataset.py")
    ap.add_argument("--split", default="sft=0.7,grpo=0.2,val=0.1", help="must match")
    ap.add_argument("--seed", type=int, default=2025, help="must match")
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda (default: cuda if available)")
    ap.add_argument("--out", default="artifacts/qc_candidates/train_candidates.csv")
    args = ap.parse_args()

    load_env(os.path.join(REPO, ".env"))
    import random

    import pandas as pd
    import torch

    from scripts.build_qc_dataset import resolve_train_csv

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(resolve_train_csv(args.train_csv))
    fracs = {k: float(v) for k, v in (kv.split("=") for kv in args.split.split(","))}
    pools = split_notes(len(df), args.n_notes, fracs, args.seed)
    in_pools = sorted(i for idx in pools.values() for i in idx)
    rng = random.Random(args.seed + 1)

    rows = []
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        if name not in REGISTRY:
            print(f"x {name}: unknown system"); continue
        kind, kw = REGISTRY[name]
        try:
            obj = _build(kind, dict(kw), device)
            _place(obj, device)
        except Exception as e:
            print(f"x {name}: load failed ({type(e).__name__}: {e})"); continue
        idx = sorted(rng.sample(in_pools, min(args.notes_per_system, len(in_pools))))
        t0 = time.perf_counter()
        for i in idx:
            note = str(df["original"][i])
            try:
                cand = obj.predict_xml(note)
            except Exception as e:                    # one bad note must not kill the run
                cand = ""
                print(f"  ! {name} note {i}: {type(e).__name__}")
            rows.append({"original": note, "candidate": cand, "source": name})
        print(f"  {name}: {len(idx)} notes in {time.perf_counter() - t0:.0f}s")
        del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
