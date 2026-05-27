import importlib


def test_all_subpackages_import():
    for module in [
        "qot_course",
        "qot_course.utils.logging_config",
        "qot_course.utils.config",
        "qot_course.hardware.runtime",
        "qot_course.summaries.build",
    ]:
        assert importlib.import_module(module) is not None
