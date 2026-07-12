"""Reproducible repo-root resolution.

Every script resolves the project root as the directory containing ``utils.py``
(kept as the root marker for backward compatibility). Import ``PROJECT_ROOT``
from here instead of hardcoding paths or repeating the resolution header.
"""
from pathlib import Path

_here = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    p for p in (_here, *_here.parents) if (p / "utils.py").is_file()
)


def project_path(*parts) -> Path:
    """Path relative to the repo root, e.g. ``project_path('main_data', 'x.csv')``."""
    return PROJECT_ROOT.joinpath(*parts)
