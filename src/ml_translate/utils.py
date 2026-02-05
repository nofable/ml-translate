import time
import math
from pathlib import Path


def get_project_root():
    """Get project root (where pyproject.toml is)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m} {s}"


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{asMinutes(s)} (- {asMinutes(rs)})"
