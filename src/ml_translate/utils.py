import logging
import math
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """Get the best available device for PyTorch.

    Auto-detects available hardware in order of preference: CUDA > MPS > CPU.

    Args:
        prefer: Optional device preference ("cuda", "mps", "cpu").
            If specified and available, uses that device.

    Returns:
        torch.device for the selected device.
    """
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return device
        elif prefer == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
            return device
        elif prefer == "cpu":
            logger.info("Using CPU")
            return torch.device("cpu")
        else:
            logger.warning(f"Preferred device '{prefer}' not available, auto-detecting")

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def get_project_root() -> Path:
    """Get project root (where pyproject.toml is)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def asMinutes(s: float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.0f}s"


def timeSince(since: float, percent: float) -> str:
    now = time.time()
    elapsed = now - since
    estimated_total = elapsed / percent
    remaining = estimated_total - elapsed
    return f"{asMinutes(elapsed)} (remaining: {asMinutes(remaining)})"
