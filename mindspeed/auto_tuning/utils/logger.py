from typing import Optional, Set
import logging
import os
from sys import stdout

_LOGGERS: Set[str] = set()
_LOG_FMT = "[%(levelname)s] %(name)s: %(message)s"
_LOG_LEVEL = logging.INFO
_LOG_FILE = None
_LOGGER_NAME_PREFIX = "auto-tuning"


def init_logger(level: str = "info", log_file: Optional[str] = None):
    global _LOG_LEVEL, _LOG_FILE
    if level == "warning":
        _LOG_LEVEL = logging.WARNING
    elif level == "debug":
        _LOG_LEVEL = logging.DEBUG
    else:
        _LOG_LEVEL = logging.INFO
    if log_file:
        _LOG_FILE = os.path.abspath(log_file)

    for name in _LOGGERS:
        logger_name = f"{_LOGGER_NAME_PREFIX}.{name}"
        logger = logging.getLogger(name=logger_name)
        logger.setLevel(_LOG_LEVEL)
        if _LOG_FILE:
            logger.addHandler(logging.FileHandler(_LOG_FILE, encoding="utf-8"))
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))


def get_logger(name: str):
    global _LOGGERS
    logger_name = f"{_LOGGER_NAME_PREFIX}.{name}"
    if name not in _LOGGERS:
        logger = logging.getLogger(name=logger_name)
        logger.propagate = False
        logger.setLevel(_LOG_LEVEL)
        logger.addHandler(logging.StreamHandler(stream=stdout))
        if _LOG_FILE:
            logger.addHandler(logging.FileHandler(_LOG_FILE, encoding="utf-8"))
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))
        _LOGGERS.add(name)
        return logger
    return logging.getLogger(logger_name)
