"""
"""

import contextlib
import os
from pathlib import Path

# Eagerly import spaces to trigger ZeroGPU mode on aokit import
try:
    import spaces
except ImportError:
    spaces = None


ARCHIVE_SO_PATTERN = '/tmp/*/archive/data/aotinductor/model/*.wrapper.so'


@contextlib.contextmanager
def register_archive_cleanup():
    """
    PyTorch already cleans-up extracted archives in /tmp
    But a GPU worker never terminates gracefully in ZeroGPU so cleanup must be done manually
    """

    if spaces is None:
        return

    try:
        from spaces.zero.utils import read_map_files
        from spaces.zero.utils import register_cleanup
    except ImportError:
        return

    pid = os.getpid()
    maps_before = {name for name, _ in read_map_files()}

    yield

    for name, path in read_map_files():
        if name not in maps_before:
            if path.match(ARCHIVE_SO_PATTERN):
                package_path = Path(*path.parts[:3])
                return register_cleanup(pid, package_path)
