import numpy as np
import torch as pt
import os 

from transition import Transition

import copy

class Policy(pt.nn.Module):
    def __init__(self, path, pathq, doubleq, lr=0.001) -> None:
        super(Policy, self).__init__()
        self.doubleq = doubleq

        if os.path.exists(path):
            self.model = pt.load(path)

            if doubleq:
                if os.path.exists(pathq):
                    self.target_model = pt.load(pathq)
                else:
                    self.target_model = copy.deepcopy(self.model)

        else:
            self.model = pt.nn.Sequential(
                pt.nn.Linear(8, 150),
                pt.nn.ReLU(),
                pt.nn.Linear(150, 120),
                pt.nn.ReLU(),
                pt.nn.Linear(120, 4),
            )
            if doubleq:
                self.target_model = pt.nn.Sequential(
                        pt.nn.Linear(8, 150),
                        pt.nn.ReLU(),
                        pt.nn.Linear(150, 120),
                        pt.nn.ReLU(),
                        pt.nn.Linear(120, 4),
                    )
            
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=lr)
        if doubleq:
            self.optimizer_prime = pt.optim.Adam(self.target_model.parameters(), lr=lr)

        self.loss = pt.nn.MSELoss()

    def select_action(self, state):
        logits = list(self.model(pt.Tensor(state)))
        return logits.index(max(logits))  # use model to get action

    def save_model(self, path="/model.pt", dqpath="/model2.pt"):
        if self.doubleq:
            pt.save(self.model, path)
            pt.save(self.target_model, dqpath)
        else:
            pt.save(self.model, path)
