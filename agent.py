import random


from memory import Memory
from policy import Policy

import torch as pt


class Agent:
    def __init__(self, epsilon, policy) -> None:
        self.memory = Memory(1000)
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

    def train(self):
        samples = self.memory.sample(32)

        for state, state_prime, reward, action in samples:
            q_prime = self.policy.forward(state_prime)
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + 0.9 * q_prime[a_prime]

            q_state = self.policy.forward(state)
            q_state[action] = a_value

            # Zero the gradients
            self.policy.optimizer.zero_grad()

            # Forward pass
            outputs_pred = self.policy.model(pt.Tensor(state))

            # Compute loss
            loss = self.policy.loss(outputs_pred, pt.Tensor(q_state))

            # Backward pass
            loss.backward()

            # Update weights
            self.policy.optimizer.step()

    def decay(self):
        self.epsilon = self.epsilon / 1.5
