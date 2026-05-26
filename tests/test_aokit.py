"""
"""

from pathlib import Path

import torch

import aokit


MLP_REPO_ID = 'aokit-tests/mlp-{device}'


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Blocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            Block(),
            Block(),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_compile_and_load(tmp_path: Path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP().to(device)
    inp = torch.randn(8, 10, device=device)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Blocks().to(device)
    inp = torch.randn(8, 10, device=device)
    out = model(inp)

    with aokit.capture(model.blocks[0]) as call:
        model(inp)
    exported = torch.export.export(model.blocks[0], call.args, call.kwargs)
    package_dir = tmp_path / 'package'
    aokit.compile_and_save(package_dir, exported, submodule='blocks')
    aokit.load_from_package_dir(model, package_dir)

    out_ = model(inp)
    assert torch.allclose(out, out_)


def test_load_from_hub():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP().to(device)
    inp = torch.randn(8, 10, device=device)
    out = model(inp)

    aokit.load(model, repo_id=MLP_REPO_ID.format(device=device))

    out_ = model(inp)
    assert torch.allclose(out, out_)
