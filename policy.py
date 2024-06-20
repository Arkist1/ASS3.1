import numpy as np
import numpy
import torch as pt

from transition import Transition


class Policy(pt.nn.Module):
    def __init__(self, path, lr=0.001) -> None:
        super(Policy, self).__init__()

        if path:
            self.model = pt.load(path)
        else:
            self.model = pt.nn.Sequential(
                pt.nn.Linear(8, 150),
                pt.nn.ReLU(),
                pt.nn.Linear(150, 120),
                pt.nn.ReLU(),
                pt.nn.Linear(120, 4),
            )

        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = pt.nn.MSELoss()

    def select_action(self, state):
        logits = list(self.model(pt.Tensor(state)))
        return logits.index(max(logits))  # use model to get action

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
