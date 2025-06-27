from typing import Set
import logging

_LOGGERS: Set[str] = set()
_LOG_FMT = "[%(levelname)s] %(name)s: %(message)s"
_LOG_LEVEL: int = logging.INFO
_LOGGER_NAME_PREFIX = "AutoTuning."


def init_logger(level: int):
    global _LOG_LEVEL, _LOG_FILE
    _LOG_LEVEL = level

    for name in _LOGGERS:
        logger_name = f"{_LOGGER_NAME_PREFIX}{name}"
        logger = logging.getLogger(name=logger_name)
        logger.setLevel(_LOG_LEVEL)
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))


def get_logger(name: str) -> logging.Logger:
    global _LOGGERS
    logger_name = f"{_LOGGER_NAME_PREFIX}{name}"
    if name not in _LOGGERS:
        logger = logging.getLogger(name=logger_name)
        logger.propagate = False
        logger.setLevel(_LOG_LEVEL)
        logger.addHandler(logging.StreamHandler())
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))
        _LOGGERS.add(name)
    return logging.getLogger(logger_name)
