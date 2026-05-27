"""
"""

import textwrap
from pathlib import Path

import pytest
import torch

import aokit

from .models import MODEL_REPO_ID
from .models import Model


def test_compile_and_load(tmp_path: Path):
    model = Model()
    inp = torch.randn(8, 10)
    out = model(inp)

    with aokit.capture(model) as call:
        model(inp)
    exported = torch.export.export(model, call.args, call.kwargs)
    package_dir = tmp_path / 'package'
    aokit.compile_and_save(package_dir, exported)
    aokit.load_from_package_dir(model, package_dir)

    out_ = model(inp)
    assert torch.allclose(out, out_)


def test_compile_and_load_repeated(tmp_path: Path):
    model = Model()
    inp = torch.randn(8, 10)
    out = model(inp)

    with aokit.capture(model.blocks[0]) as call:
        model(inp)
    exported = torch.export.export(model.blocks[0], call.args, call.kwargs)
    package_dir = tmp_path / 'package'
    aokit.compile_and_save(package_dir, exported, submodule='blocks')
    aokit.load_from_package_dir(model, package_dir)

    out_ = model(inp)
    assert torch.allclose(out, out_)


@pytest.mark.parametrize('use_loader', [False, True])
def test_load_from_hub(use_loader: bool):
    model = Model()
    inp = torch.randn(8, 10)
    out = model(inp)

    def loader(module: torch.nn.Module, package_dir: str):
        aokit.load_from_package_dir(module, package_dir)

    if use_loader:
        aokit.load(model, repo_id=MODEL_REPO_ID, aokit_loader=loader)
    else:
        aokit.load(model, repo_id=MODEL_REPO_ID)

    out_ = model(inp)
    assert torch.allclose(out, out_)


def test_load_source_code():
    assert aokit.load_source_code('model', repo_id='repo/id') == textwrap.dedent(
        f"""\
            aokit.load(
                module=model,
                repo_id='repo/id',
            )
        """
    ).rstrip()
