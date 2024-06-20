import numpy as np
import numpy
import torch as pt

from transition import Transition


class Policy(pt.nn.Module):
    def __init__(self, path, lr=0.01) -> None:
        super(Policy, self).__init__()

        if path:
            self.model = pt.load(path)
        else:
            self.model = pt.nn.Sequential(
                pt.nn.Linear(8, 512),
                pt.nn.ReLU(),
                pt.nn.Linear(512, 512),
                pt.nn.ReLU(),
                pt.nn.Linear(512, 4),
            ).to(device="cuda")

        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = pt.nn.MSELoss()

    def forward(self, state):
        state = pt.Tensor(state).to(device="cuda")
        return list(self.model(state))

    def select_action(self, state):
        logits = self.forward(state)
        return logits.index(max(logits))  # use model to get action

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
