"""
Fork utils
"""

import multiprocessing
import os
import shutil
from pathlib import Path


def maybe_create_queue():
    try:
        ctx = multiprocessing.get_context('fork')
    except ValueError:  # pragma: no cover
        return
    return ctx.Queue()


def remove_after_wait(pid: int, path: Path):
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:  # pragma: no cover
        pass
    shutil.rmtree(path, ignore_errors=True)
