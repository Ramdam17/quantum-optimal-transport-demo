import qot_course.hardware.runtime as rt
from qiskit_aer import AerSimulator


def test_default_backend_is_local_simulator():
    backend = rt.get_backend()
    assert isinstance(backend, AerSimulator)


def test_prefer_hardware_falls_back_when_service_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(rt, "_make_runtime_service", _raise)
    backend = rt.get_backend(prefer_hardware=True)
    assert isinstance(backend, AerSimulator)  # graceful fallback


def test_noisy_backend_is_runnable():
    backend = rt.get_noisy_backend()
    assert hasattr(backend, "run")
