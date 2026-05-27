from qot_course.utils.config import deep_merge, load_config


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\npaths:\n  out: results\n")
    cfg = load_config(p)
    assert cfg["seed"] == 7
    assert cfg["paths"]["out"] == "results"


def test_deep_merge_overrides_nested_keys():
    base = {"seed": 42, "paths": {"a": 1, "b": 2}}
    override = {"paths": {"b": 99}}
    merged = deep_merge(base, override)
    assert merged == {"seed": 42, "paths": {"a": 1, "b": 99}}
    assert base["paths"]["b"] == 2  # input not mutated
