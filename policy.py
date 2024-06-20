import numpy as np
import torch as pt

from transition import Transition


class Policy:
    def __init__(self) -> None:
        self.model = 1  # make model here

    def select_move(self, state):
        # return self.model.calculate(state)  # use model to get action
        return 1

    def save_model(self, path="/model.pt"):
        pt.save(self.model, path)
