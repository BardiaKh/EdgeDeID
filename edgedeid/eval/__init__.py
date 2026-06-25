"""Evaluation utilities for PHI de-identification."""

from .metrics import EvalResult, TagMetrics, bootstrap_ci, compute_metrics, extract_entities, print_report, to_dataframe
from .inference import batch_inference, run_vllm_model
