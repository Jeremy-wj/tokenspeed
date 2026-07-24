"""Shared AR+RMSNorm benchmark shape defaults."""
from __future__ import annotations

import os


MODEL_RESIDUAL_WIDTHS = {
    "gpt_oss_120b": 2880,
    "glm_4_6": 5120,
    "deepseek_v3": 7168,
    "compressed_kv": 512,
    "compressed_q": 1536,
}

DEFAULT_N_VALUES = [512, 1536, 2880, 5120, 7168]


def default_hidden_size() -> int:
    """Resolve an explicit benchmark/model hidden size, then the regression fixture."""
    return int(os.environ.get("BENCH_N", os.environ.get("HIDDEN_SIZE", "2880")))

