"""
"""

import torch


MODEL_REPO_ID = 'aokit-tests/Model-cpu'


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
        x = self.relu(x)
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            Block(),
            Block(),
        ])
        self.fc = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
