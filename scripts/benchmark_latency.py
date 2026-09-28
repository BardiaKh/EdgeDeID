#!/usr/bin/env python3
"""Single-note latency benchmark with a guaranteed common device.

What was wrong with the notebook benchmark (baselines.ipynb §8)
----------------------------------------------------------------
1. No common device. It set ``CUDA_VISIBLE_DEVICES=""`` mid-kernel, after §4
   had already initialized CUDA, so the variable had no effect. Only models
   exposing ``.model`` (ours) were force-moved to CPU in float32. Transformers
   pipelines (RoBERTa, BERT, OpenMed, Stanford) default to ``cuda:0`` when CUDA
   is available (transformers 5.x: ``device=None -> 0``); OPF picks CUDA
   explicitly; GLiNER, Presidio and scispaCy stay on CPU. Evidence: the "CPU"
   latencies of the pipeline baselines match their batched-GPU §4 timings to
   within a few ms (RoBERTa 12.3 vs 12.9 ms). The published gap (23 s/note vs
   ~10 ms) compared our model on CPU fp32 against GPU baselines.
2. "Peak RSS" was the *difference* in current RSS before/after loading, inside
   one long-lived process. Freed memory is reused rather than returned to the
   OS, so later models show ~0 or negative values (-1 MB appears in the table).
3. The GGUF cell spawned ``llama-cli`` per note, which reloads the model every
   call and times model loading, not inference. (It was never run.)

What this script does instead
-----------------------------
* Every model runs in a FRESH subprocess whose environment is fixed before
  Python starts: ``--device cpu`` hides all GPUs, so nothing can use one.
* After loading, the model is placed on the requested device and the device it
  actually runs on is read back from its parameters and reported per row
  (``device_verified``). A row whose device does not match is flagged.
* Peak RSS is the process high-water mark (``ru_maxrss`` / ``peak_wset``),
  reported absolute and net of the post-import baseline. Peak GPU memory from
  ``torch.cuda.max_memory_allocated``.
* For our generative models it also records prompt and output token counts,
  prefill time (one forward pass over the prompt ~ time-to-first-token), and
  decode throughput — so the number is interpretable: on CPU, decoding ~180
  output tokens dominates; the 867-token instruction block is ~82% of every
  prompt but prefill is the smaller share.
* Optional ``--llama-server URL``: times a GGUF model served ONCE by
  llama.cpp's ``llama-server`` over HTTP (load-once; the deployment path).

Generative vs encoder latency is structurally different — an encoder labels a
note in one forward pass, a generator writes ~180 tokens one at a time — and
this script makes that visible rather than hiding it.

Usage
-----
    # CPU table (edge-realistic), all models:
    python scripts/benchmark_latency.py --device cpu --notes-from ablation/B_predictions.csv
    # GPU table:
    python scripts/benchmark_latency.py --device cuda --notes-from ablation/B_predictions.csv
    # a subset, fewer notes:
    python scripts/benchmark_latency.py --device cpu --models EdgeDeID-SFT,RoBERTa-i2b2 --n 10
    # quantized deployment runtime (start the server yourself first):
    #   llama-server -m edgedeid-sft-Q4_K_M.gguf -c 8192 -t 8 --port 8080
    python scripts/benchmark_latency.py --device cpu --models GGUF \
        --llama-server http://127.0.0.1:8080 --gguf-label EdgeDeID-SFT-Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# name -> (kind, constructor args). Mirrors baselines.ipynb BASELINES + OUR_CHECKPOINTS.
REGISTRY: dict[str, tuple[str, dict]] = {
    "Presidio":        ("presidio", {"score_threshold": 0.5}),
    "GLiNER":          ("gliner", {"model_name": "urchade/gliner_mediumv2.1", "threshold": 0.5}),
    "RoBERTa-i2b2":    ("roberta", {"model_name": "obi/deid_roberta_i2b2"}),
    "BERT-i2b2":       ("bert", {"model_name": "obi/deid_bert_i2b2"}),
    "OpenMed-PII":     ("openmed", {"model_name": "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1"}),
    "Stanford-deid":   ("stanford", {"model_name": "StanfordAIMI/stanford-deidentifier-base"}),
    "scispaCy":        ("scispacy", {"model_name": "en_core_web_lg"}),
    "OPF":             ("opf", {}),
    "Qwen3-0.6B-Base": ("qwen", {"model_src": "Qwen/Qwen3-0.6B-Base"}),
    "EdgeDeID-SFT":    ("qwen", {"model_src": "EdgeDeID/Qwen3-0.6B-Base_SFT"}),
    "EdgeDeID-GRPO":   ("qwen", {"model_src": "EdgeDeID/Qwen3-0.6B-GRPO"}),
}
DEFAULT_MODELS = [m for m in REGISTRY]


# ─── child: load one model, time it, print one JSON line ──────────────────────

def _build(kind: str, kw: dict, device: str):
    from edgedeid import baselines as B

    if kind == "presidio":
        return B.PresidioBaseline(**kw)
    if kind == "gliner":
        return B.GLiNERBaseline(**kw)
    if kind == "roberta":
        return B.RobertaDeidBaseline(**kw)
    if kind == "bert":
        return B.BertDeidBaseline(**kw)
    if kind == "openmed":
        return B.OpenMedPIIBaseline(**kw)
    if kind == "stanford":
        return B.StanfordDeidBaseline(**kw)
    if kind == "scispacy":
        return B.ScispaCyBaseline(**kw)
    if kind == "opf":
        return B.OPFBaseline(device=device, **kw)
    if kind == "qwen":
        return B.QwenDeidBaseline(name=kw["model_src"], **kw)
    raise ValueError(kind)


def _torch_modules(obj):
    """Torch modules inside a baseline wrapper, wherever it keeps them.

    Checks the known attributes (``model``, ``_model``, ``_pipe.model``), then
    scans one level into opaque holders (``_opf``, a non-Module ``_model``) —
    bounded, so it cannot wander through a whole object graph.
    """
    import torch

    found, seen = [], set()

    def add(kind, attr, m):
        if isinstance(m, torch.nn.Module) and id(m) not in seen:
            seen.add(id(m))
            found.append((kind, attr, m))

    for attr in ("model", "_model"):
        m = getattr(obj, attr, None)
        add("attr", attr, m)
        if m is not None and not isinstance(m, torch.nn.Module):
            add("inner", attr, getattr(m, "model", None))
    pipe = getattr(obj, "_pipe", None)
    if pipe is not None:
        add("pipe", "_pipe", getattr(pipe, "model", None))
    for holder_name in ("_opf", "_model"):
        holder = getattr(obj, holder_name, None)
        if holder is not None and not isinstance(holder, torch.nn.Module):
            for k, v in list(vars(holder).items()) if hasattr(holder, "__dict__") else []:
                add("scan", f"{holder_name}.{k}", v)
    return found


def _place(obj, device: str) -> None:
    """Move every torch module the wrapper holds onto ``device``."""
    import torch

    for kind, attr, mod in _torch_modules(obj):
        mod.to(device)
        if kind == "pipe":
            obj._pipe.device = torch.device(device if device != "cuda" else "cuda:0")
    if hasattr(obj, "device") and not callable(getattr(obj, "device")):
        try:
            obj.device = device
        except Exception:
            pass


def _verified_device(obj, requested: str) -> str:
    mods = _torch_modules(obj)
    if not mods and hasattr(obj, "_opf"):
        # model not reachable from outside; the device was passed explicitly, and
        # with --device cpu no GPU is visible to this process at all
        return f"{requested} (passed to OPF)"
    if not mods:
        return "cpu (no torch module: spaCy/regex)"
    devs = set()
    for _, _, m in mods:
        try:
            devs.add(str(next(m.parameters()).device).split(":")[0])
        except StopIteration:
            pass
    return ",".join(sorted(devs)) or "unknown"


def _peak_rss_mb() -> float | None:
    try:
        import resource                                   # Linux/macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 if sys.platform != "darwin" else r / 1e6   # KB on Linux, B on mac
    except ImportError:
        pass
    try:
        import psutil                                     # Windows
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 1e6
    except Exception:
        return None


def _cur_rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def _pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def child_main(args) -> None:
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    notes = _load_notes(args.notes_from, args.n + args.warmup)
    rss_base = _cur_rss_mb()
    device = args.device
    row: dict = {"model": args.model, "device_requested": device}

    if args.model == "GGUF":
        predict, extra = _gguf_predictor(args), {}
        row.update(runtime="llama.cpp server", device_verified="cpu (llama.cpp)")
    else:
        kind, kw = REGISTRY[args.model]
        kw = dict(kw)
        if args.model in args.overrides:
            kw["model_src" if kind == "qwen" else "model_name"] = args.overrides[args.model]
        t_load = time.perf_counter()
        obj = _build(kind, kw, device)
        _place(obj, device)
        row["load_s"] = round(time.perf_counter() - t_load, 2)
        row["device_verified"] = _verified_device(obj, device)
        row["runtime"] = "transformers" if kind != "opf" else "opf"
        if kind == "qwen":
            row["dtype"] = str(next(obj.model.parameters()).dtype).replace("torch.", "")
        predict = obj.predict_xml
        extra = {"qwen": obj} if kind == "qwen" else {}

    dv = row["device_verified"]
    cpu_only_tool = dv.startswith("cpu (no torch") or dv.startswith("cpu (llama")
    row["device_ok"] = dv.startswith(device) or cpu_only_tool
    if device == "cuda" and cpu_only_tool:
        row["note"] = "no GPU path (spaCy/regex/llama.cpp CPU); report as CPU"

    sync = torch.cuda.synchronize if device == "cuda" and torch.cuda.is_available() else (lambda: None)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for n in notes[: args.warmup]:
        predict(n)
    ms, outs = [], []
    for n in notes[args.warmup: args.warmup + args.n]:
        sync(); t0 = time.perf_counter()
        outs.append(predict(n))
        sync(); ms.append((time.perf_counter() - t0) * 1000)

    row.update({
        "n_notes": len(ms),
        "ms/note (median)": round(statistics.median(ms), 1),
        "ms/note (p90)": round(_pctl(ms, 0.9), 1),
        "ms/note (mean)": round(statistics.mean(ms), 1),
        "notes/sec": round(1000 / statistics.median(ms), 3),
        "peak_rss_mb": round(_peak_rss_mb() or 0),
        "rss_after_imports_mb": round(rss_base or 0),
        "torch_threads": torch.get_num_threads(),
    })
    row["peak_rss_net_mb"] = row["peak_rss_mb"] - row["rss_after_imports_mb"]
    if device == "cuda" and torch.cuda.is_available():
        row["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
        row["gpu"] = torch.cuda.get_device_name(0)

    q = extra.get("qwen")
    if q is not None:                     # interpretability for generative models
        from edgedeid.eval.inference import _build_prompt
        tok = q.tokenizer
        p_tok, o_tok, pre = [], [], []
        for n, o in zip(notes[args.warmup: args.warmup + args.n], outs):
            enc = tok(_build_prompt(n), return_tensors="pt").to(q.device)
            p_tok.append(enc["input_ids"].shape[-1])
            o_tok.append(len(tok(o, add_special_tokens=False)["input_ids"]))
            with torch.no_grad():
                sync(); t0 = time.perf_counter()
                q.model(**enc, use_cache=True)
                sync(); pre.append((time.perf_counter() - t0) * 1000)
        dec_ms = [max(1e-6, m - p) for m, p in zip(ms, pre)]
        row.update({
            "prompt_tokens (median)": statistics.median(p_tok),
            "output_tokens (median)": statistics.median(o_tok),
            "prefill_ms (median)": round(statistics.median(pre), 1),
            "decode_tok/s (median)": round(statistics.median(
                [o / (d / 1000) for o, d in zip(o_tok, dec_ms)]), 1),
        })

    import transformers
    row["versions"] = f"torch {torch.__version__} | transformers {transformers.__version__}"
    print("RESULT_JSON " + json.dumps(row))


def _gguf_predictor(args):
    import urllib.request

    from edgedeid.eval.inference import _build_prompt

    url = args.llama_server.rstrip("/") + "/completion"

    def predict(note: str) -> str:
        body = json.dumps({"prompt": _build_prompt(note), "n_predict": 1024,
                           "temperature": 0, "stop": ["</phi_entities>"],
                           "cache_prompt": args.cache_prompt}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return "<phi_entities>\n" + json.loads(r.read())["content"] + "</phi_entities>"
    return predict


def _load_notes(path: str, k: int) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    col = next((c for c in ("note", "text", "original") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{path}: need a note/text/original column")
    notes = df[col].fillna("").astype(str).tolist()
    if len(notes) < k:
        raise SystemExit(f"{path}: {len(notes)} notes, need {k} (n + warmup)")
    return notes[:k]


# ─── parent: one subprocess per model ─────────────────────────────────────────

def _hardware(device: str) -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cpu": platform.processor() or platform.machine()}
    try:
        with open("/proc/cpuinfo") as f:
            name = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), None)
        if name:
            info["cpu"] = name
    except OSError:
        pass
    try:
        import psutil
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    if device == "cuda":
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            info["gpu"] = out.stdout.strip()
        except Exception:
            pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help=f"comma list from {list(REGISTRY)} (+ GGUF)")
    ap.add_argument("--notes-from", default="ablation/B_predictions.csv",
                    help="CSV with a note/text column (the radiology benchmark notes)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads (default: torch's choice; recorded)")
    ap.add_argument("--override", action="append", default=[],
                    help="NAME=hf_id_or_local_path, e.g. EdgeDeID-SFT=/ckpt/sft")
    ap.add_argument("--timeout-min", type=float, default=120)
    ap.add_argument("--llama-server", default=None)
    ap.add_argument("--gguf-label", default="EdgeDeID-GGUF")
    ap.add_argument("--cache-prompt", action="store_true",
                    help="let llama-server reuse the KV of the shared instruction prefix")
    ap.add_argument("--out-dir", default="artifacts/latency")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.overrides = dict(o.split("=", 1) for o in args.override)

    # HF_TOKEN for the private EdgeDeID/* checkpoints; children inherit the env.
    from edgedeid.qc.train_utils import load_env
    load_env(os.path.join(REPO, ".env"))

    if args.child:
        child_main(args)
        return

    import pandas as pd

    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""          # set BEFORE the child's Python starts
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name != "GGUF" and name not in REGISTRY:
            print(f"x {name}: unknown model"); continue
        if name == "GGUF" and not args.llama_server:
            print("x GGUF: needs --llama-server"); continue
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--model", name,
               "--device", args.device, "--notes-from", args.notes_from,
               "--n", str(args.n), "--warmup", str(args.warmup)]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        for o in args.override:
            cmd += ["--override", o]
        if name == "GGUF":
            cmd += ["--llama-server", args.llama_server]
            if args.cache_prompt:
                cmd.append("--cache-prompt")
        print(f"> {name} ({args.device}) ...", flush=True)
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=args.timeout_min * 60, cwd=REPO)
        except subprocess.TimeoutExpired:
            print(f"  x timed out after {args.timeout_min} min"); continue
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if line is None:
            tail = "\n".join((p.stderr or p.stdout).strip().splitlines()[-6:])
            print(f"  x failed (exit {p.returncode}):\n{tail}"); continue
        row = json.loads(line[len("RESULT_JSON "):])
        if name == "GGUF":
            row["model"] = args.gguf_label
        flag = "" if row.get("device_ok") else "   <-- DEVICE MISMATCH"
        print(f"  {row['ms/note (median)']} ms/note (p90 {row['ms/note (p90)']}) on "
              f"{row['device_verified']}, peak RSS {row['peak_rss_mb']} MB{flag}")
        rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"latency_{args.device}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = {"hardware": _hardware(args.device), "args": {k: v for k, v in vars(args).items()
                                                        if k not in ("child", "model")}}
    with open(os.path.join(args.out_dir, f"latency_{args.device}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {out} (+ _meta.json with hardware)")
    bad = [r["model"] for r in rows if not r.get("device_ok")]
    if bad:
        print(f"WARNING: not on {args.device}: {bad} — do not put these in the same table")


if __name__ == "__main__":
    main()
