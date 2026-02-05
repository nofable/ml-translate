import math
import time
from pathlib import Path


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
    return f"{m} {s}"


def timeSince(since: float, percent: float) -> str:
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{asMinutes(s)} (- {asMinutes(rs)})"
