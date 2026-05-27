"""
"""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def map_files_diff():
    maps_before = {name for name, _ in _read_map_files()}
    maps_diff: set[Path] = set()
    yield maps_diff
    for name, path in _read_map_files():
        if name not in maps_before:
            maps_diff.add(path)


def _read_map_files():
    for map_file in Path('/proc/self/map_files').iterdir():
        try:
            path = map_file.readlink()
        except OSError: # pragma: no cover
            continue
        yield map_file.name, path
