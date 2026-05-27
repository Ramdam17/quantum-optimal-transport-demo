import logging

from qot_course.utils.logging_config import get_logger


def test_get_logger_returns_named_logger_with_handler():
    logger = get_logger("qot_course.test", level="DEBUG")
    assert logger.name == "qot_course.test"
    assert logger.level == logging.DEBUG
    assert logger.handlers, "logger should have at least one handler"


def test_get_logger_is_idempotent():
    a = get_logger("qot_course.dup")
    b = get_logger("qot_course.dup")
    assert a is b
    assert len(a.handlers) == len(b.handlers)  # no duplicate handlers
