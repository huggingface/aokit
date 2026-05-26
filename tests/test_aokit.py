"""
"""

from pathlib import Path

import torch

import aokit


def test_compile_and_load(tmp_path: Path):

    class Model(torch.nn.Module):
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
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
