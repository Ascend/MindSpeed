from typing import Set
import logging
from sys import stdout

_LOGGERS: Set[str] = set()
_LOG_FMT = "[%(levelname)s] %(name)s: %(message)s"
_LOGGER_DEFAULT_HANDLER = logging.StreamHandler(stream=stdout)
_LOG_LEVEL: int = logging.INFO
_LOGGER_NAME_PREFIX = "AutoSettings."


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
        # Only add the default handler if no handlers exist
        if not logger.handlers:
            logger.addHandler(_LOGGER_DEFAULT_HANDLER)
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))
        _LOGGERS.add(name)
    return logging.getLogger(logger_name)


def change_stream_handler(logger, new_stream, old_stream='default'):
    # Remove all existing stream handlers
    for handler in logger.handlers[:]:  # Create a copy of the list to iterate over
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
    
    # Add the new stream handler
    new_handler = logging.StreamHandler(stream=new_stream)
    new_handler.setFormatter(logging.Formatter(fmt=_LOG_FMT))
    logger.addHandler(new_handler)
    
    return logger

