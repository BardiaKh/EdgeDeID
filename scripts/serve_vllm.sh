#!/usr/bin/env bash
# Serve ONE checkpoint with vLLM on the OpenAI-compatible API.
#
# Usage:
#   ./serve_vllm.sh <model> [port] [gpu]
# Example:
#   ./serve_vllm.sh EdgeDeID/Qwen3-0.6B-Base_SFT 9995 0
#
# Leave it running, then point the notebook §5 (or eval_one_vllm.py) at the port.
set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <model> [port] [gpu]}"
PORT="${2:-9995}"
GPU="${3:-0}"

VLLM_USE_V1="0" CUDA_VISIBLE_DEVICES="${GPU}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --generation-config vllm \
    --dtype bfloat16
