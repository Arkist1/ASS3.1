import numpy as np
import torch as pt

from transition import Transition


class Policy(pt.nn.Module):
    def __init__(self) -> None:
        super(Policy, self).__init__()
        self.flatten = pt.nn.Flatten()
        self.model = pt.nn.Sequential(
            pt.nn.Linear(8, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 4),
        )

    def forward(self, state):
        state = pt.Tensor(state)
        return list(self.model(state))

    def select_action(self, state):
        logits = self.forward(state)
        return logits.index(max(logits))  # use model to get action

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
