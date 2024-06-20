from transition import Transition
from collections import deque
import random

import torch
import numpy as np


class Memory:
    def __init__(self, capacity):
        self.transition_deque = deque(maxlen=capacity)

    def store(self, transition: Transition):
        self.transition_deque.append(transition)
        return

    def sample(self, batch_size=64):
        if len(self.transition_deque) < batch_size:
            return self.transition_deque
        batch = random.sample(list(self.transition_deque), batch_size)
        return batch
