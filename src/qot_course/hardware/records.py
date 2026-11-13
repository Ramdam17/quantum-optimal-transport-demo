"""Keep the raw counts of every backend run as one JSON record per run.

A number quoted from a quantum device can only be recomputed if the counts it was computed from
are kept, together with the job identifier that lets the provider find the job again. This module
writes those facts to disk, one small JSON file per run, so that every run of a module-05
notebook, offline or on a real QPU, leaves its counts behind.

The record holds the device name, the date and time of the run (UTC), the job identifier when
there is one, the number of shots, and the raw counts. Offline records are overwritten by the
next offline run of the same notebook and tag, so re-executing a notebook does not pile up
files; a real run carries its job identifier in the file name and is never overwritten.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Environment variable that overrides the folder the records are written to.
RESULTS_DIR_ENV = "QOT_HARDWARE_RESULTS_DIR"

#: Default folder, relative to the repository root (``src/qot_course/hardware`` is three levels
#: below it).
DEFAULT_RESULTS_SUBDIR = Path("results") / "hardware"


def default_results_dir() -> Path:
    """Return the folder that run records go to.

    Returns
    -------
    pathlib.Path
        ``$QOT_HARDWARE_RESULTS_DIR`` when that variable is set, otherwise
        ``<repository root>/results/hardware``.
    """
    override = os.environ.get(RESULTS_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / DEFAULT_RESULTS_SUBDIR


def _setting_key(key: Any) -> str:
    """Turn a counts-dictionary key into a JSON key (a tomography setting tuple becomes ``"XY"``)."""
    if isinstance(key, tuple):
        return "".join(str(part) for part in key)
    return str(key)


def _slug(text: str) -> str:
    """Return ``text`` with every character outside ``[A-Za-z0-9_-]`` replaced by ``-``."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def save_run_record(
    counts: Mapping[Any, Any],
    *,
    notebook: str,
    device: str,
    shots: int,
    job_id: str | None = None,
    is_real: bool = False,
    tag: str = "run",
    results_dir: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the raw counts of one run, with its device, date, shots and job id, to a JSON file.

    Parameters
    ----------
    counts : Mapping
        The raw counts. Either one counts dictionary (bitstring -> int), or a mapping from a
        measurement setting to a counts dictionary (tomography). A setting key that is a tuple,
        such as ``("X", "Y")``, is written as the string ``"XY"``.
    notebook : str
        Short identifier of the notebook that ran the job, e.g. ``"05_02"``.
    device : str
        Name of the backend that ran the job (``backend.name``).
    shots : int
        Shots per circuit.
    job_id : str or None
        Provider job identifier. Pass it for a real QPU run; ``None`` offline.
    is_real : bool
        ``True`` only when the job ran on a real QPU.
    tag : str
        Name of the run inside the notebook (``"bell"``, ``"calibration"``, ``"k08_rep3"``...),
        so that several runs of one notebook do not overwrite each other.
    results_dir : str, Path or None
        Folder to write to; ``None`` uses :func:`default_results_dir`.
    extra : Mapping or None
        Further JSON-serialisable fields to store (a seed, a circuit description...).

    Returns
    -------
    pathlib.Path
        Path of the file written. Its name is
        ``<notebook>_<tag>__<device>__<job_id or "offline">.json``.

    When to use
    -----------
    In the run cell of every module-05 notebook, right after the result comes back, so that a
    value later quoted in prose can be recomputed from the counts.

    Examples
    --------
    >>> import tempfile
    >>> folder = tempfile.mkdtemp()
    >>> path = save_run_record({"0": 2040, "1": 2056}, notebook="05_02", device="fake",
    ...                        shots=4096, results_dir=folder)
    >>> path.name
    '05_02_run__fake__offline.json'
    """
    folder = Path(results_dir) if results_dir is not None else default_results_dir()
    folder.mkdir(parents=True, exist_ok=True)

    values = list(counts.values())
    if values and all(isinstance(v, Mapping) for v in values):
        raw = {
            _setting_key(k): {str(b): int(c) for b, c in v.items()}
            for k, v in counts.items()
        }
    else:
        raw = {str(b): int(c) for b, c in counts.items()}

    record: dict[str, Any] = {
        "notebook": notebook,
        "tag": tag,
        "device": device,
        "is_real": bool(is_real),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "job_id": job_id,
        "shots": int(shots),
        "counts": raw,
    }
    if extra:
        record["extra"] = dict(extra)

    name = f"{_slug(notebook)}_{_slug(tag)}__{_slug(device)}__{_slug(job_id) if job_id else 'offline'}.json"
    path = folder / name
    path.write_text(
        json.dumps(record, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    logger.info("Run record written: %s", path)
    return path
