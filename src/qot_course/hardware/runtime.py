"""Quantum backend selection with graceful fallback to a local simulator.

The course is simulator-first: by default everything runs on AerSimulator,
offline and reproducibly. Real IBM hardware is opt-in and degrades gracefully
to a simulator when no credentials are available.

Real-hardware idioms (validated against the owner's Qiskit 2.x / runtime 0.47
tutorials) live in the session notebooks: ``QiskitRuntimeService().least_busy``,
``generate_preset_pass_manager``, and ``SamplerV2``/``EstimatorV2``.
"""

from __future__ import annotations

import os

from qiskit_aer import AerSimulator

from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)


def _make_runtime_service():
    """Create an IBM Qiskit Runtime service (separated for testability).

    If ``IBM_QUANTUM_TOKEN`` is set in the environment (e.g. a git-ignored
    ``.env``) it is passed explicitly; otherwise the account saved on disk by
    ``QiskitRuntimeService.save_account`` is read with no arguments. The channel
    defaults to ``ibm_quantum_platform`` (current IBM Quantum Platform; the
    legacy ``ibm_quantum`` channel was retired mid-2025) and can be overridden
    with ``IBM_QUANTUM_CHANNEL``.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if token:
        channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
        return QiskitRuntimeService(channel=channel, token=token)
    return QiskitRuntimeService()


def get_backend(prefer_hardware: bool = False):
    """Return a quantum backend.

    With ``prefer_hardware=True``, try the least-busy real QPU; on any failure
    (e.g. missing credentials) fall back to a local ``AerSimulator``.
    """
    if prefer_hardware:
        try:
            service = _make_runtime_service()
            return service.least_busy(operational=True, simulator=False)
        except Exception as exc:  # noqa: BLE001 - fall back on anything
            logger.warning("Hardware unavailable, using simulator: %s", exc)
    return AerSimulator()


def get_noisy_backend(fake: str = "FakeManilaV2"):
    """Return an ``AerSimulator`` carrying a real device's noise model.

    Falls back to a noiseless simulator if the fake backend cannot be loaded.
    """
    try:
        from qiskit_ibm_runtime import fake_provider

        fake_backend = getattr(fake_provider, fake)()
        return AerSimulator.from_backend(fake_backend)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Fake backend %s unavailable, using ideal simulator: %s", fake, exc
        )
        return AerSimulator()


def select_backend(use_hardware: bool = False, fake: str = "FakeManilaV2"):
    """Select a quantum backend and report, honestly, what was chosen.

    Parameters
    ----------
    use_hardware : bool
        If ``True``, request the least-busy real IBM QPU (credentials via
        ``save_account`` or ``IBM_QUANTUM_TOKEN``); on any failure, fall back to
        a local simulator and report ``is_real=False``. If ``False`` (default),
        return an offline ``AerSimulator`` carrying ``fake``'s noise model.
    fake : str
        Name of the fake backend whose noise model to load offline.

    Returns
    -------
    backend : qiskit BackendV2
        Usable as ``SamplerV2(mode=backend)`` (local testing mode when offline).
    label : str
        Human-readable description of where the cell actually ran.
    is_real : bool
        ``True`` only if a real QPU was obtained.

    Examples
    --------
    >>> backend, label, is_real = select_backend(use_hardware=False)
    >>> is_real
    False
    """
    if use_hardware:
        backend = get_backend(prefer_hardware=True)
        if not isinstance(backend, AerSimulator):
            return backend, f"{backend.name} (real IBM QPU)", True
        return backend, "AerSimulator (ideal — hardware unavailable, check credentials)", False
    backend = get_noisy_backend(fake)
    return backend, f"AerSimulator[{fake} noise model] (offline)", False
