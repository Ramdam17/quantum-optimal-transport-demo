import json

from qot_course.hardware.records import (
    RESULTS_DIR_ENV,
    default_results_dir,
    save_run_record,
)


def test_flat_counts_record_round_trips(tmp_path):
    path = save_run_record(
        {"0": 2040, "1": 2056},
        notebook="05_02",
        device="aer_sim",
        shots=4096,
        results_dir=tmp_path,
    )
    assert path.name == "05_02_run__aer_sim__offline.json"
    record = json.loads(path.read_text())
    assert record["counts"] == {"0": 2040, "1": 2056}
    assert record["shots"] == 4096
    assert record["job_id"] is None
    assert record["is_real"] is False
    assert record["device"] == "aer_sim"
    assert record["date_utc"].endswith("+00:00")


def test_tomography_settings_keys_become_strings(tmp_path):
    counts = {("X", "Y"): {"00": 3, "11": 5}, ("Z", "Z"): {"01": 8}}
    path = save_run_record(
        counts, notebook="05_04", device="d", shots=8, tag="bell", results_dir=tmp_path
    )
    record = json.loads(path.read_text())
    assert record["counts"] == {"XY": {"00": 3, "11": 5}, "ZZ": {"01": 8}}


def test_real_run_carries_job_id_in_name(tmp_path):
    path = save_run_record(
        {"0": 1},
        notebook="05_01",
        device="ibm_marrakesh",
        shots=1,
        job_id="abc123",
        is_real=True,
        results_dir=tmp_path,
        extra={"seed": None},
    )
    assert path.name == "05_01_run__ibm_marrakesh__abc123.json"
    record = json.loads(path.read_text())
    assert record["job_id"] == "abc123" and record["is_real"] is True
    assert record["extra"] == {"seed": None}


def test_device_name_is_slugged(tmp_path):
    path = save_run_record(
        {"0": 1},
        notebook="05_01",
        device="aer_simulator_from(fake_marrakesh)",
        shots=1,
        results_dir=tmp_path,
    )
    assert path.name == "05_01_run__aer_simulator_from-fake_marrakesh__offline.json"


def test_results_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path / "elsewhere"))
    assert default_results_dir() == tmp_path / "elsewhere"
    monkeypatch.delenv(RESULTS_DIR_ENV)
    assert default_results_dir().parts[-2:] == ("results", "hardware")
