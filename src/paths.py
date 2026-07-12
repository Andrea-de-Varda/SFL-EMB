"""Reproducible repo-root resolution.

``src/paths.py`` lives at ``<repo>/src/paths.py``, so the repo root is its
grandparent. Import ``PROJECT_ROOT`` / ``project_path`` from here instead of
hardcoding paths or repeating a resolution header in every script.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts) -> Path:
    """Path relative to the repo root, e.g. ``project_path('main_data', 'x.csv')``."""
    return PROJECT_ROOT.joinpath(*parts)
