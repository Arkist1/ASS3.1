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

    def select_action(self, state):
        state = pt.Tensor(state)
        logits = list(self.model(state))
        action = logits.index(max(logits))
        return action  # use model to get action

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
