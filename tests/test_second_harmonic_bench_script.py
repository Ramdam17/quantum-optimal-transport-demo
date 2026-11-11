import csv
import json

from scripts.run_second_harmonic_bench import main


def _rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_quick_run_writes_every_result_file(tmp_path):
    assert main(["--quick", "--out", str(tmp_path)]) == 0
    for name in ("map.csv", "freedom.csv", "continuity.json", "timestep_check.csv", "comparisons.csv",
                 "summary.csv", "reliability.csv", "run_metadata.json"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert len(_rows(tmp_path / "map.csv")) == 4 * 5
    comp = _rows(tmp_path / "comparisons.csv")
    assert {r["experiment"] for r in comp} == {"detection", "equal_plv", "equal_moduli", "trap"}
    assert {r["statistic"] for r in comp} >= {"plv", "plv2", "qmi_rich", "bures_rich", "qmi_weighted_02", "qmi_weighted_05"}
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["config"]["protocol"]["n_records"] == 6 and meta["comparisons_not_made"] == []
    cont = json.loads((tmp_path / "continuity.json").read_text())
    assert cont["spearman_abs_m1_abs_m2"] == 1.0


def test_an_unreachable_equal_moduli_pair_is_reported_not_hidden(tmp_path):
    cfg = tmp_path / "c.yaml"
    base = (__import__("pathlib").Path(__file__).parents[1] / "config" / "base.yaml").read_text()
    cfg.write_text(base.replace("k2_ref: 0.12", "k2_ref: 0.15"))
    assert main(["--quick", "--config", str(cfg), "--out", str(tmp_path / "out")]) == 0
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["comparisons_not_made"][0]["experiment"] == "equal_moduli"
    assert "closest" in meta["comparisons_not_made"][0]["reason"]
    assert "equal_moduli" not in {r["experiment"] for r in _rows(tmp_path / "out" / "comparisons.csv")}
