import random
import numpy as np

from memory import Memory
from policy import Policy

import torch as pt


class Agent:
    def __init__(self, epsilon, policy) -> None:
        self.memory = Memory(100_000)
        self.policy = Policy(policy)

        self.moves = [0, 1, 2, 3]

        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.choice(self.moves)
        else:
            return self.policy.select_action(state)

    def store_transition(self, transition):
        self.memory.store(transition)

    def train(self, learning_rate):
        samples = self.memory.sample(10)
        if len(samples) < 10:
            return

        X = []
        Y = []

        for state, state_prime, action, reward, terminal in samples:
            q_prime = list(self.policy.model(pt.Tensor(state_prime)))
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + (0.99 * q_prime[a_prime]) * (1 - terminal)

            q_state = self.policy.model(pt.Tensor(state))
            X.append(q_state)

            q_state = list(q_state)
            q_state[action] = a_value

            Y.append(q_state)

        # Forward pass
        # outputs_pred = self.policy.model(pt.Tensor(np.array(X)))

        # Compute loss
        loss = self.policy.loss(pt.stack(X), pt.Tensor(Y))
        # print(loss)

        # Zero the gradients
        self.policy.optimizer.zero_grad()

        # Backward pass
        loss.backward()
        self.policy.optimizer.step()


    def decay(self):
        self.epsilon = self.epsilon * 0.996
