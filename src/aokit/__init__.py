"""
AOKit
"""

import importlib.metadata

from . import exporting
from .aokit import compile_and_save
from .aokit import load
from .aokit import load_from_package_dir
from .aokit import load_from_module_dir
from .aokit import load_source_code
from .aokit import patch
from .aokit import LazyAOTIModel
from .aokit import LazyAOTIModelWithWeights


__all__ = [
    'exporting',
    'compile_and_save',
    'load',
    'load_from_package_dir',
    'load_from_module_dir',
    'load_source_code',
    'patch',
    'LazyAOTIModel',
    'LazyAOTIModelWithWeights',
]


try:
    __version__ = importlib.metadata.version('aokit')
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '0+unknown'
