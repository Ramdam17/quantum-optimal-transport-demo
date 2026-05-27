"""Quantum backend selection with graceful fallback to a local simulator.

The course is simulator-first: by default everything runs on AerSimulator,
offline and reproducibly. Real IBM hardware is opt-in and degrades gracefully
to a simulator when no credentials are available.

Real-hardware idioms (validated against the owner's Qiskit 2.x / runtime 0.47
tutorials) live in the session notebooks: ``QiskitRuntimeService().least_busy``,
``generate_preset_pass_manager``, and ``SamplerV2``/``EstimatorV2``.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator

from qot_course.utils.logging_config import get_logger

logger = get_logger(__name__)


def _make_runtime_service():
    """Create an IBM Qiskit Runtime service (separated for testability)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

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
