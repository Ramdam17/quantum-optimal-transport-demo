"""Check the configuration loader, on a toy file and on the file the course ships."""

import inspect
from pathlib import Path

from qot_course.quantum_ot.capstone import sweep_coupling_measures
from qot_course.utils.config import deep_merge, load_config

BASE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "base.yaml"


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\npaths:\n  out: results\n")
    cfg = load_config(p)
    assert cfg["seed"] == 7
    assert cfg["paths"]["out"] == "results"


def test_load_config_returns_an_empty_dict_for_an_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(p) == {}


def test_deep_merge_overrides_nested_keys():
    base = {"seed": 42, "paths": {"a": 1, "b": 2}}
    override = {"paths": {"b": 99}}
    merged = deep_merge(base, override)
    assert merged == {"seed": 42, "paths": {"a": 1, "b": 99}}
    assert base["paths"]["b"] == 2  # input not mutated


def test_base_config_has_the_sections_the_code_reads():
    cfg = load_config(BASE_CONFIG)
    assert isinstance(cfg["seed"], int)
    assert {"capstone", "second_harmonic_bench"} <= set(cfg)
    capstone = cfg["capstone"]
    assert 0.0 < capstone["example_duration_s"] <= capstone["sweep_duration_s"]


def test_base_config_and_the_capstone_sweep_state_the_same_defaults():
    """The sweep's docstring sends the reader to ``config/base.yaml`` for its defaults."""
    capstone = load_config(BASE_CONFIG)["capstone"]
    defaults = inspect.signature(sweep_coupling_measures).parameters
    assert defaults["duration"].default == capstone["sweep_duration_s"]
    assert defaults["seed"].default == capstone["sweep_seed"]
