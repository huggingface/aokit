"""
"""

import multiprocessing
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch

import aokit

from .models import Model


@pytest.mark.parametrize('sigkill', [False, True])
def test_fork_cleanup(tmp_path: Path, sigkill: bool):
    model = Model()
    inp = torch.randn(8, 10)

    with aokit.capture(model) as call:
        model(inp)
    exported = torch.export.export(model, call.args, call.kwargs)
    package_dir = tmp_path / 'package'
    aokit.compile_and_save(package_dir, exported)
    aokit.load_from_package_dir(model, package_dir)

    mp_context = multiprocessing.get_context('fork')
    ready = mp_context.Event()

    def target(sleep: float):
        model(inp)
        ready.set()
        time.sleep(sleep)

    process = mp_context.Process(target=target, args=((2 if sigkill else 1),))
    process.start()
    ready.wait()

    if sigkill:
        assert process.pid is not None
        os.kill(process.pid, signal.SIGKILL)
    process.join()
