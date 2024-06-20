import random
import numpy as np

from memory import Memory
from policy import Policy

import torch as pt


class Agent:
    def __init__(self, memory_size, sample_size, epsilon, discount, lr, policy) -> None:
        self.memory = Memory(memory_size)
        self.policy = Policy(policy, lr=lr)

        self.moves = [0, 1, 2, 3]

        self.epsilon = epsilon
        self.discount = discount
        self.sample_size = sample_size
        
        self.memory_filled = False

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.choice(self.moves)
        else:
            return self.policy.select_action(state)

    def store_transition(self, transition):
        self.memory.store(transition)

    def train(self):
        if not self.memory_filled:
            if not len(self.memory.transition_deque) > self.sample_size:
                return
            self.memory_filled = True

        samples = self.memory.sample(self.sample_size)

        X = []
        Y = []

        for state, state_prime, action, reward, terminal in samples:
            q_prime = list(self.policy.model(pt.Tensor(state_prime)))
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + (self.discount * q_prime[a_prime]) * (1 - terminal)

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
