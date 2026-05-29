"""Root conftest.py — adds the project root to sys.path so that scripts/
is importable from tests (e.g. ``from scripts.show_palette import ...``).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
