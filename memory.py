from transition import Transition
from collections import deque
import random


class Memory:
    def __init__(self, capacity):
        self.transition_deque = deque()
        self.capacity = capacity

    def store(self, transition: Transition):
        if len(self.transition_deque) >= self.capacity:
            self.transition_deque.popleft()
        self.transition_deque.append(transition)
        return

    def sample(self, batch_size=100):
        if len(self.transition_deque) < batch_size:
            return self.transition_deque
        batch = random.sample(list(self.transition_deque), batch_size)
        return batch
