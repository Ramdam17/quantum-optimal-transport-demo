import qot_course.hardware.runtime as rt
import qiskit_ibm_runtime
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


def test_make_runtime_service_uses_env_token(monkeypatch):
    """A token in IBM_QUANTUM_TOKEN is passed explicitly (the .env path)."""
    captured = {}
    monkeypatch.setattr(
        qiskit_ibm_runtime,
        "QiskitRuntimeService",
        lambda **kw: captured.update(kw) or "SVC",
    )
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "TESTTOKEN")
    out = rt._make_runtime_service()
    assert out == "SVC"
    assert captured == {"channel": "ibm_quantum_platform", "token": "TESTTOKEN"}


def test_make_runtime_service_without_env_token_reads_saved_account(monkeypatch):
    """With no env token, fall back to a no-arg service (the save_account path)."""
    captured = {}
    monkeypatch.setattr(
        qiskit_ibm_runtime,
        "QiskitRuntimeService",
        lambda **kw: captured.update(kw) or "SVC",
    )
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    rt._make_runtime_service()
    assert captured == {}  # QiskitRuntimeService() with no kwargs


def test_select_backend_offline_is_noisy_sim_and_honest(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    backend, label, is_real = rt.select_backend(use_hardware=False)
    assert isinstance(backend, AerSimulator)
    assert is_real is False
    assert "noise model" in label


def test_select_backend_hardware_without_creds_falls_back_honestly(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(rt, "_make_runtime_service", _raise)
    backend, label, is_real = rt.select_backend(use_hardware=True)
    assert isinstance(backend, AerSimulator)  # graceful fallback
    assert is_real is False  # and honest about it


def test_select_backend_offline_default_is_the_marrakesh_twin(monkeypatch):
    """Module 05 quotes ibm_marrakesh (Heron, cz); its offline twin must be that device's model."""
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    backend, label, is_real = rt.select_backend(use_hardware=False)
    assert "FakeMarrakesh" in label
    assert "cz" in backend.operation_names
    assert is_real is False


def test_noisy_backend_default_stays_manila_for_module_01():
    """01/11 and 01/15 read FakeManilaV2 calibration for the qubit this backend uses."""
    assert rt.get_noisy_backend().num_qubits == 5
