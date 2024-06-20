import numpy as np
import torch as pt

from transition import Transition


class Policy(pt.nn.Module):
    def __init__(self) -> None:
        super(Policy, self).__init__()
        self.model = 1  # make model here

        self.flatten = pt.nn.Flatten()
        self.layers = pt.nn.Sequential(
            pt.nn.Linear(9, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 4),
        )

    def select_action(self, state):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        print(logits)
        return max(logits)  # use model to get action

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
