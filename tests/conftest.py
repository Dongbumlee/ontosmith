"""Test bootstrap: the staging root is the import root.

The pipeline package is imported as ``pipeline`` (its public
layout name); tests run from a bare checkout with no install
step, so the repo root goes on sys.path here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
