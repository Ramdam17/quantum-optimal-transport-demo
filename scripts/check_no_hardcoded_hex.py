"""Guard: forbid hardcoded #rrggbb literals in new-style notebooks.

Colours must come from qot_course.colors. Legacy notebooks/s*.ipynb are exempt
(migrated later). Run:  uv run python scripts/check_no_hardcoded_hex.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
ROOT = Path(__file__).resolve().parents[1]
NEW_NOTEBOOK_GLOB = "notebooks/[0-9][0-9]_*/[0-9][0-9]_*.ipynb"


def find_hardcoded_hex() -> list[str]:
    offenders: list[str] = []
    for nb_path in sorted(ROOT.glob(NEW_NOTEBOOK_GLOB)):
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for line in cell.get("source", []):
                if HEX.search(line):
                    offenders.append(f"{nb_path.relative_to(ROOT)}: {line.strip()}")
    return offenders


if __name__ == "__main__":
    found = find_hardcoded_hex()
    if found:
        print("Hardcoded hex (use qot_course.colors instead):")
        for line in found:
            print("  " + line)
        raise SystemExit(1)
    print("OK — no hardcoded hex in new notebooks.")
