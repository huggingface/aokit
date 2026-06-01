"""
AOKit
"""

import importlib.metadata
import inspect
import multiprocessing
import os
import textwrap
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any
from typing import Callable
from typing import cast

import torch
import torch._inductor.codecache  # https://github.com/pytorch/pytorch/pull/165157
from packaging import version
from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters
from torch._inductor.cpu_vec_isa import valid_vec_isa_list
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights

from ._internal import elf
from ._internal import fork
from ._internal import proc


EXTERNAL_WEIGHTS_AOTI_CONFIGS: dict[str, Any] = {
    'aot_inductor.package_constants_in_so': False,
    'always_keep_tensor_constants': True,
    'joint_graph_constant_folding': False,
    **(
        {'aot_inductor.package_constants_on_disk_format': "pickle_weights"}
        if version.parse(version.parse(torch.__version__).base_version) >= version.parse('2.10')
        else {'aot_inductor.package_constants_on_disk': True}
    ),
}


AOKIT_PACKAGE_ROOT = 'package'
AOKIT_PACKAGE_NAME = 'package.pt2'


def compile_and_save(
    package_dir: str | os.PathLike[str],
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
    submodule: str | None = None,
) -> None:
    """Compile an exported program with AOTInductor and write the package to disk.

    The compiled artifacts are bundled into a ``.pt2`` archive (with weights kept
    external) and stored under ``package_dir``. When ``submodule`` is given the
    archive is placed under ``submodules/<submodule>``; otherwise it goes under
    ``root``.

    Args:
        package_dir: Directory under which the package archive is written.
        exported_program: The exported program to compile.
        inductor_configs: Optional extra TorchInductor configs, merged with the
            external-weights configs required by AOKit.
        submodule: Optional submodule name to namespace the package under.
    """
    archive_file = BytesIO()
    files, _weights = _compile(exported_program, inductor_configs)
    package_aoti(archive_file, list(files))
    if submodule is not None:
        subdir_path = Path(package_dir) / 'submodules' / submodule
    else:
        subdir_path = Path(package_dir) / 'root'
    subdir_path.mkdir(parents=True, exist_ok=True)
    package_path = subdir_path / AOKIT_PACKAGE_NAME
    package_path.write_bytes(archive_file.getbuffer())


def _compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
):
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs
    inductor_configs = {**(inductor_configs or {}), **EXTERNAL_WEIGHTS_AOTI_CONFIGS}
    aot_compile_options = {**inductor_configs, 'aot_inductor.package': True}
    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=aot_compile_options)
    artifacts = cast(list[str | Weights], artifacts)
    files = [file for file in artifacts if isinstance(file, str)]
    for file in files:
        if file.endswith('.so'):
            elf.clear_execstack(Path(file))
    (weights,) = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    return files, weights


class LazyAOTIModel:
    def __init__(self, archive_file: torch.types.FileLike):
        self.archive_file = archive_file
        self.compiled_model: ContextVar[AOTICompiledModel | None] = ContextVar('compiled_model', default=None)
        self.loaded_weights: ContextVar[dict[str, torch.Tensor] | None] = ContextVar('loaded_weights', default=None)
        self._init_pid = os.getpid()
        self._cleanup_queue: multiprocessing.Queue[tuple[int, Path] | None] | None = fork.maybe_create_queue()
        if self._cleanup_queue is not None:
            Thread(target=self._cleanup_thread_target, name="AOKit-Cleanup", daemon=True).start()
        valid_vec_isa_list()  # pre-warm functools.cache

    def _cleanup_thread_target(self):
        assert self._cleanup_queue is not None
        while (item := self._cleanup_queue.get()) is not None:
            pid, path = item
            Thread(target=fork.remove_after_wait, args=(pid, path), name=f"AOKit-Cleanup-{pid}", daemon=True).start()

    def load_package(self):
        """Load the AOTI package and return the compiled model.

        In the parent process the package is loaded directly. In a forked child
        process, the loaded ``.so`` is tracked so it can be cleaned up once the
        owning process exits (see :meth:`_cleanup_thread_target`).
        """
        if self._cleanup_queue is None or (pid := os.getpid()) == self._init_pid:
            return torch._inductor.aoti_load_package(self.archive_file)
        with proc.map_files_diff() as mapped_paths:
            res = torch._inductor.aoti_load_package(self.archive_file)
        for path in mapped_paths:
            if path.match('/tmp/*/archive/data/aotinductor/*/*.so'):
                self._cleanup_queue.put((pid, Path(*path.parts[:3])))
                break
        return res

    def __call__(self, weights: dict[str, torch.Tensor], check_full_update: bool, *args, **kwargs):
        """Run the compiled model, lazily loading the package and weights.

        The compiled model and the currently loaded weights are cached per
        context. The package is loaded on first use, and constants are
        (re)loaded whenever the provided ``weights`` differ from the cached ones.

        Args:
            weights: Mapping of constant FQN to tensor; only entries matching the
                model's constant FQNs are loaded.
            check_full_update: Whether to verify that all constants are updated
                when loading constants.
            *args: Positional inputs forwarded to the compiled model.
            **kwargs: Keyword inputs forwarded to the compiled model.

        Returns:
            The output of the compiled model.
        """
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

    def __del__(self):  # pragma: no cover
        if self._cleanup_queue is not None:
            self._cleanup_queue.put(None)

    def with_weights(self, weights: dict[str, torch.Tensor]):
        """Bind a set of weights to this model.

        Returns a :class:`LazyAOTIModelWithWeights` that calls this model with the
        given ``weights``, so it can be used as a drop-in ``forward`` replacement.
        """
        return LazyAOTIModelWithWeights(self, weights)


class LazyAOTIModelWithWeights:
    def __init__(self, model: LazyAOTIModel, weights: dict[str, torch.Tensor]):
        self.model = model
        self.weights = weights
        self.first_call = True

    def __call__(self, *args, **kwargs):
        """Invoke the underlying model with the bound weights.

        On the first call ``check_full_update`` is enabled to verify that all
        constants are updated; subsequent calls skip that check.
        """
        check_full_update = self.first_call
        self.first_call = False
        return self.model(self.weights, check_full_update, *args, **kwargs)


def load(
    module: torch.nn.Module,
    repo_id: str,
    revision: str | None = None,
    aokit_loader: Callable[[torch.nn.Module, str], Any] | None = None,
):
    """Download an AOKit package from the Hugging Face Hub and load it into a module.

    Fetches the ``package/`` directory from the given repository and patches the
    module's submodules (and/or root module) with the compiled AOTI models.

    Args:
        module: The module to patch with the compiled package.
        repo_id: Hugging Face Hub repository id to download from.
        revision: Optional git revision (branch, tag, or commit) to download.
        aokit_loader: Optional custom loader called with ``(module, package_dir)``;
            defaults to :func:`load_from_package_dir`.
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=f'{AOKIT_PACKAGE_ROOT}/*',
    )
    package_dir = Path(repo_path) / AOKIT_PACKAGE_ROOT
    if aokit_loader is not None:
        aokit_loader(module, str(package_dir))
    else:
        load_from_package_dir(module, package_dir)


def load_source_code(
    module_expr: str,
    repo_id: str,
    revision: str | None = None,
    aokit_loader: Callable[[torch.nn.Module, str], Any] | None = None,
) -> str:
    """Generate Python source code that reproduces an :func:`load` call.

    Builds a snippet that loads the package for ``module_expr`` from the given
    repository, inlining the source of ``aokit_loader`` when one is provided.
    Arguments left as ``None`` are omitted from the generated call.

    Args:
        module_expr: Source expression evaluating to the module to load into.
        repo_id: Hugging Face Hub repository id to load from.
        revision: Optional git revision to pin in the generated call.
        aokit_loader: Optional custom loader whose source is inlined and
            referenced by name in the generated call.

    Returns:
        A string of Python source code for the ``aokit.load(...)`` call.
    """
    loader_name = aokit_loader.__name__ if aokit_loader is not None else 'None'
    loader_source = ""
    if aokit_loader is not None:  # pragma: no cover
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
    """Load all compiled packages found in a package directory into a module.

    Loads each compiled submodule from ``package_dir/submodules/<name>`` into the
    corresponding submodule, and the root package from ``package_dir/root`` into
    the module itself. Missing directories are skipped.

    Args:
        module: The module (or module list) to patch.
        package_dir: Directory containing ``submodules/`` and/or ``root/``.
    """
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
) -> None:
    """Load a single compiled package and patch it onto a module.

    Wraps the package in a :class:`LazyAOTIModel`. If ``module`` is a
    :class:`~torch.nn.ModuleList`, every block is patched with the same compiled
    model; otherwise the module itself is patched.

    Args:
        module: The module (or module list) to patch.
        module_dir: Directory containing the package archive.
    """
    module_dir = Path(module_dir)
    aoti_model = LazyAOTIModel(module_dir / AOKIT_PACKAGE_NAME)
    if isinstance(module, torch.nn.ModuleList):
        for block in module:
            patch(block, aoti_model)
    else:
        patch(module, aoti_model)


def patch(
    module: torch.nn.Module,
    aoti_model: LazyAOTIModel,
    call_method: str = 'forward',
) -> None:
    """Replace a module's call method with a compiled AOTI model.

    Shallow-clones the module to avoid mutating the original, unwraps any tensor
    subclass parameters, binds the clone's ``state_dict`` to ``aoti_model``, and
    sets the result as ``module.<call_method>``.

    Args:
        module: The module whose method is replaced.
        aoti_model: The compiled model to bind the weights to.
        call_method: Name of the attribute to override (defaults to ``'forward'``).
    """
    module_ = _shallow_clone_module(module)  # Prevent original module mutation
    unwrap_tensor_subclass_parameters(module_)  # https://github.com/pytorch/pytorch/issues/159918
    aoti_model_with_weights = aoti_model.with_weights(module_.state_dict())
    setattr(module, call_method, aoti_model_with_weights)


def _shallow_clone_module(module: torch.nn.Module) -> torch.nn.Module:
    clone = object.__new__(module.__class__)
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = {k: _shallow_clone_module(v) for k, v in module._modules.items() if v is not None}
    return clone


try:
    __version__ = importlib.metadata.version('aokit')
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '0+unknown'
