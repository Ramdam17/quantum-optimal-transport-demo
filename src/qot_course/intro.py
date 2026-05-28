"""Session 1 helpers: a small environment check.

Plotting lives in :mod:`qot_course.viz` (reused across the whole course).
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata

_KEY_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pot",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "cvxpy",
]


def check_environment() -> dict[str, str | None]:
    """Return installed versions of the course's key packages.

    Maps each package name to its version string, or ``None`` if missing.
    """
    versions: dict[str, str | None] = {}
    for package in _KEY_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions
