"""Shared runtime settings for the analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"


def _ensure_dir(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


os.environ.setdefault("MPLCONFIGDIR", _ensure_dir(CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", _ensure_dir(CACHE_DIR / "numba"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # Matplotlib is optional for non-plotting imports; plotting scripts will
    # surface any real import error when they import pyplot.
    pass
