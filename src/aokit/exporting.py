"""
AOKit exporting tools
"""

import contextlib
from typing import Any
from typing import Callable
from unittest import mock

import torch


@contextlib.contextmanager
def capture(
    module: torch.nn.Module | Callable[..., Any],
    call_method: str = 'forward',
):
    class CapturedCallException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    class CapturedCall:
        def __init__(self):
            self.args: tuple[Any, ...] = ()
            self.kwargs: dict[str, Any] = {}

    captured_call = CapturedCall()

    def capture_call(*args, **kwargs):
        raise CapturedCallException(*args, **kwargs)

    with mock.patch.object(module, call_method, new=capture_call):
        try:
            yield captured_call
        except CapturedCallException as e:
            captured_call.args = e.args
            captured_call.kwargs = e.kwargs
