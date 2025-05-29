"""Handle the transition from megatron_adaptor to megatron_v2."""

from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from logging import getLogger
from typing import Callable, Iterable, Optional

LOG = getLogger(__name__)

ENABLE_DEPRECATED = False

# when decide to deprecate megatron_adaptor_deprecated.py, define a specific time.
MEGATRON_ADAPTOR_DEPRECATED_TIME = datetime(2099, 1, 1, tzinfo=timezone.utc)


class DeprecatedLogLevel(Enum):
    """Define log level when decorate a deprecated function interface."""

    CODE = 1
    FUNCTION = 2
    MODULE = 3


class DeprecatedError(Exception):
    """Raise this exception when a deprecated function
    is called after deprecated date.
    """

    def __init__(self, deprecated_date: datetime, func: str):
        super().__init__(f"Function {func} was deprecated on {deprecated_date}")
        self._deprecated_date = deprecated_date
        self._func = func

    def __str__(self):

        return f"""function {self._func} is deprecated
                    at {self._deprecated_date}, but today is {datetime.now}.
                    if you still wanna use the deprecated function,
                    please set `mindspeed.ENABLE_DEPRECATED=True`,
                    but it won't be sure all function works well.
                """


class Deprecated:
    """A decorator for functions which will be deprecated.

    example:
        ```python
        @Deprecated(
            deprecated_date=MEGATRON_ADAPTOR_DEPRECATED_TIME,
            deprecated_codes=(
                "TransformerBlock._build_layers = _build_layers",
                "aspm.register_patch('megatron.training.training.num_floating_point_operations', num_floating_point_wrapper)",
                "aspm.register_patch('megatron.core.transformer.moe.moe_utils.track_moe_metrics', track_moe_metrics)",
            ),
            suggestion=
            "please use mindspeed.megatron_adaptor.py"
            "instead of interface in mindspeed.megatron_adaptor_deprecated.py",
        )
        def mcore_models_adaptation(aspm, mindspeed_args):
            # code here
        ```
    """

    def __init__(
        self,
        deprecated_date: datetime,
        deprecated_codes: Optional[Iterable[str]] = None,
        log_level: DeprecatedLogLevel = DeprecatedLogLevel.FUNCTION,
        suggestion: str = "",
    ):
        self._deprecated_date = deprecated_date
        self._deprecated_codes = deprecated_codes
        self._log_level = log_level
        self._suggestion = suggestion

    def __call__(self, func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._is_deprecated():
                raise DeprecatedError(self._deprecated_date, func.__name__)

            self._add_warning_log(func=func)
            ret = func(*args, **kwargs)

            return ret

        return wrapper

    def _add_warning_log(self, func: Callable):
        if self._log_level == DeprecatedLogLevel.MODULE:
            LOG.warning(
                "module of %s will deprecated after %s, Suggestion: %s",
                func.__module__,
                self._deprecated_date,
                self._suggestion,
            )
        elif self._log_level == DeprecatedLogLevel.FUNCTION:
            LOG.warning(
                "function %s in %s will deprecated after %s, Suggestion: %s",
                func.__name__,
                func.__module__,
                self._deprecated_date,
                self._suggestion,
            )
        elif self._deprecated_codes:
            for code in self._deprecated_codes:
                LOG.warning(
                    "code %s of function %s in module %s "
                    "will deprecated after %s, Suggestion: %s",
                    code,
                    func.__name__,
                    func.__module__,
                    self._deprecated_date,
                    self._suggestion,
                )

    def _is_deprecated(self) -> bool:
        return (
            not ENABLE_DEPRECATED  # type: ignore
            and datetime.now(timezone.utc) > self._deprecated_date
        )


class AutoExecuteFunction:
    AUTO_EXECUTE = True

    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kwargs):
        if not AutoExecuteFunction.AUTO_EXECUTE:
            return None
        return self._func(*args, **kwargs)


class NoExecuteFunction:
    def __enter__(self):
        AutoExecuteFunction.AUTO_EXECUTE = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        AutoExecuteFunction.AUTO_EXECUTE = True
