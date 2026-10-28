#!/usr/bin/env bash
# Evaluate all three of our Qwen checkpoints on en_data.csv, unattended.
#
# For each checkpoint it: starts a vLLM server, waits until the API is healthy,
# runs eval_one_vllm.py, then shuts the server down before the next one.
# Results accumulate in results/comparison_ours.csv (merge with the baselines
# table in baselines.ipynb §6, or just concat the two CSVs).
#
# Usage:
#   ./run_our_vllm.sh [port] [gpu]
set -euo pipefail

PORT="${1:-9995}"
GPU="${2:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# (HF id | display name) — point the first column at local paths if you have them.
MODELS=(
  "Qwen/Qwen3-0.6B-Base|Qwen3-0.6B-Base"
  "EdgeDeID/Qwen3-0.6B-Base_SFT|EdgeDeID-SFT"
  "EdgeDeID/Qwen3-0.6B-GRPO-KL|EdgeDeID-GRPO"
)

wait_for_server() {  # poll /v1/models until ready (max ~5 min)
  for _ in $(seq 1 150); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "ERROR: vLLM server did not become ready on port ${PORT}" >&2
  return 1
}

for entry in "${MODELS[@]}"; do
  MODEL="${entry%%|*}"
  NAME="${entry##*|}"
  echo "=============================================================="
  echo ">> Serving ${NAME}  (${MODEL})"
  echo "=============================================================="

  VLLM_USE_V1="0" CUDA_VISIBLE_DEVICES="${GPU}" \
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL}" --port "${PORT}" \
      --tensor-parallel-size 1 --max-model-len 4096 \
      --generation-config vllm --dtype bfloat16 \
      > "/tmp/vllm_${NAME}.log" 2>&1 &
  SERVER_PID=$!

  # Ensure the server is killed even if eval fails.
  trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

  if wait_for_server; then
    python "${HERE}/eval_one_vllm.py" --model "${MODEL}" --name "${NAME}" --port "${PORT}"
  fi

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  trap - EXIT
  sleep 3   # let the GPU free before the next model
done

echo "All checkpoints done -> artifacts/baselines/comparison_ours.csv"
