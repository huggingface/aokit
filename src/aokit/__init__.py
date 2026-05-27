"""
"""

import contextlib
import inspect
import multiprocessing
import os
import textwrap
from collections.abc import Iterable
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any
from typing import Callable
from typing import cast
from unittest import mock

import torch
import torch._inductor.codecache # https://github.com/pytorch/pytorch/pull/165157
from packaging import version
from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters
from torch._inductor.cpu_vec_isa import valid_vec_isa_list
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights

from . import _elf
from . import _fork
from . import _proc


INDUCTOR_CONFIGS_OVERRIDES: dict[str, Any] = {
    'aot_inductor.package_constants_in_so': False,
    'aot_inductor.package_constants_on_disk': True,
    'aot_inductor.package': True,
    'always_keep_tensor_constants': True,
    'joint_graph_constant_folding': False,
}

if version.parse(version.parse(torch.__version__).base_version) >= version.parse('2.10'): # pragma: no cover
    del INDUCTOR_CONFIGS_OVERRIDES['aot_inductor.package_constants_on_disk']
    INDUCTOR_CONFIGS_OVERRIDES['aot_inductor.package_constants_on_disk_format'] = "pickle_weights"

PACKAGE_DIRNAME = 'package'
PACKAGE_FILENAME = 'package.pt2'


def compile_and_save(
    package_dir: str | os.PathLike[str],
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
    submodule: str | None = None,
):
    archive_file = BytesIO()
    files, _weights = _compile(exported_program, inductor_configs)
    package_aoti(archive_file, list(files))
    if submodule is not None:
        subdir_path = Path(package_dir) / 'submodules' / submodule
    else:
        subdir_path = Path(package_dir) / 'root'
    subdir_path.mkdir(parents=True, exist_ok=True)
    package_path = subdir_path / PACKAGE_FILENAME
    package_path.write_bytes(archive_file.getbuffer())


def _compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
):
    inductor_configs = {**(inductor_configs or {}), **INDUCTOR_CONFIGS_OVERRIDES}
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs
    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=inductor_configs) # pyright: ignore [reportArgumentType]
    artifacts = cast(list[str | Weights], artifacts)
    files = [file for file in artifacts if isinstance(file, str)]
    for file in files:
        if file.endswith('.so'):
            _elf.clear_execstack(Path(file))
    weights, = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    return files, weights


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


class LazyAOTIModel:
    def __init__(self, archive_file: torch.types.FileLike):
        self.archive_file = archive_file
        self.compiled_model: ContextVar[AOTICompiledModel | None] = ContextVar('compiled_model', default=None)
        self.loaded_weights: ContextVar[dict[str, torch.Tensor] | None] = ContextVar('loaded_weights', default=None)
        self._init_pid = os.getpid()
        self._cleanup_queue: multiprocessing.Queue[tuple[int, Path] | None] | None = _fork.maybe_create_queue()
        if self._cleanup_queue is not None:
            Thread(target=self._cleanup_thread_target, name="AOKit-Cleanup", daemon=True).start()
        valid_vec_isa_list() # pre-warm functools.cache

    def _cleanup_thread_target(self):
        assert self._cleanup_queue is not None
        while (item := self._cleanup_queue.get()) is not None:
            pid, path = item
            Thread(target=_fork.remove_after_wait, args=(pid, path), name=f"AOKit-Cleanup-{pid}", daemon=True).start()

    def load_package(self):
        if self._cleanup_queue is None or (pid := os.getpid()) == self._init_pid:
            return torch._inductor.aoti_load_package(self.archive_file)
        with _proc.map_files_diff() as mapped_paths:
            res = torch._inductor.aoti_load_package(self.archive_file)
        for path in mapped_paths:
            if path.match('/tmp/*/archive/data/aotinductor/*/*.so'):
                self._cleanup_queue.put((pid, Path(*path.parts[:3])))
                break
        return res

    def __call__(self, weights: dict[str, torch.Tensor], check_full_update: bool, *args, **kwargs):
        if (compiled_model := self.compiled_model.get()) is None:
            compiled_model = self.load_package()
            compiled_model = cast(AOTICompiledModel, compiled_model)
            self.compiled_model.set(compiled_model)
        if (loaded_weights := self.loaded_weights.get()) is None or loaded_weights is not weights:
            constant_fqns = compiled_model.get_constant_fqns()
            constant_map = {name: tensor for name, tensor in weights.items() if name in constant_fqns}
            # TODO: Explicit warn on missing / unexpected fqns
            compiled_model.load_constants(constant_map, check_full_update=check_full_update, user_managed=True)
            self.loaded_weights.set(weights)
        return compiled_model(*args, **kwargs)

    def __del__(self): # pragma: no cover
        if self._cleanup_queue is not None:
            self._cleanup_queue.put(None)

    def with_weights(self, weights: dict[str, torch.Tensor]):
        return LazyAOTIModelWithWeights(self, weights)


class LazyAOTIModelWithWeights:
    def __init__(self, model: LazyAOTIModel, weights: dict[str, torch.Tensor]):
        self.model = model
        self.weights = weights
        self.first_call = True

    def __call__(self, *args, **kwargs):
        check_full_update = self.first_call
        self.first_call = False
        return self.model(self.weights, check_full_update, *args, **kwargs)


def load(
    module: torch.nn.Module,
    repo_id: str,
    revision: str | None = None,
    aokit_loader: Callable[[torch.nn.Module, str], Any] | None = None,
):
    from huggingface_hub import snapshot_download
    repo_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=f'{PACKAGE_DIRNAME}/*',
    )
    package_dir = Path(repo_path) / PACKAGE_DIRNAME
    if aokit_loader is not None:
        aokit_loader(module, str(package_dir))
    else:
        load_from_package_dir(module, package_dir)


def load_source_code(
    module_expr: str,
    repo_id: str,
    revision: str | None = None,
    aokit_loader: Callable[[torch.nn.Module, str], Any] | None = None,
):
    loader_name = aokit_loader.__name__ if aokit_loader is not None else 'None'
    loader_source = ""
    if aokit_loader is not None: # pragma: no cover
        loader_source += textwrap.dedent(inspect.getsource(aokit_loader)).strip()
        loader_source += "\n\n"
    load_source = textwrap.dedent(
        f"""\
            aokit.load(
                module={module_expr},
                repo_id={repo_id!r},
                revision={revision!r},
                aokit_loader={loader_name},
            )
        """
    )
    load_source = '\n'.join(line for line in load_source.splitlines() if "=None," not in line)
    return loader_source + load_source


def load_from_package_dir(
    module: torch.nn.Module | torch.nn.ModuleList,
    package_dir: str | os.PathLike[str],
):
    # Structure
    package_dir = Path(package_dir)
    submodules_dir = package_dir / 'submodules'
    rootmodule_dir = package_dir / 'root'
    # Submodules
    if submodules_dir.is_dir():
        for subpackage_dir in submodules_dir.iterdir():
            if subpackage_dir.is_dir():
                submodule = module.get_submodule(subpackage_dir.name)
                load_from_module_dir(submodule, subpackage_dir)
    # Root module
    if rootmodule_dir.is_dir():
        load_from_module_dir(module, rootmodule_dir)


def load_from_module_dir(
    module: torch.nn.Module | torch.nn.ModuleList,
    module_dir: str | os.PathLike[str],
):
    module_dir = Path(module_dir)
    aoti_model = LazyAOTIModel(module_dir / PACKAGE_FILENAME)
    if isinstance(module, Iterable):
        for block in module:
            patch(block, aoti_model)
    else:
        patch(module, aoti_model)


def patch(
    module: torch.nn.Module,
    aoti_model: LazyAOTIModel,
    call_method: str = 'forward',
):
    module_ = _shallow_clone_module(module) # Prevent original module mutation
    unwrap_tensor_subclass_parameters(module_) # https://github.com/pytorch/pytorch/issues/159918
    aoti_model_with_weights = aoti_model.with_weights(module_.state_dict())
    setattr(module, call_method, aoti_model_with_weights)


def _shallow_clone_module(module: torch.nn.Module) -> torch.nn.Module:
    clone = object.__new__(module.__class__)
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = {k: _shallow_clone_module(v) for k, v in module._modules.items() if v is not None}
    return clone
