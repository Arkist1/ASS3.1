import random


from memory import Memory
from policy import Policy


class Agent:
    def __init__(self, epsilon) -> None:
        self.memory = Memory()
        self.policy = Policy()
        self.moves = [0, 1, 2, 3]

        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.choice(self.moves)
        else:
            self.policy.select_action()

    def train(self):
        samples = self.memory.sample(16)

        for state, state_prime, reward, action in samples:
            pass  # do stuff

        s_prime = self.select_action()

    def decay(self):
        self.epsilon = self.epsilon / 1.5
