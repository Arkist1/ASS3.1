import random
import numpy as np

from memory import Memory
from policy import Policy

import torch as pt


class Agent:
    def __init__(self, epsilon, policy) -> None:
        self.memory = Memory(10_000)
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

        states = []
        states_p = []
        rewards = []
        actions = []
        terminals = []
        for state, state_prime, action, reward, terminal in samples:
            states.append(state)
            states_p.append(state_prime)
            rewards.append(reward)
            actions.append(action)
            terminals.append(terminal)

            # if state_prime[6] == 1 and state_prime[7] == 1:
            #     a_value = reward
            # else:
            # q_target = self.policy.forward(state_prime).detach().max()# .detach().max(1)[0].unsqueeze(1)
            # a_prime = q_prime.index(max(q_prime))
            # a_value = reward + learning_rate * (0.99 * q_prime[a_prime]) * (1 - terminal)

            # q_state = self.policy.forward(state)

            # # q_state[action] = (a_value - q_state[action])**2
            # q_state[action] = a_value
            # X.append(state)  # value state
            # Y.append(q_state)  # q_targets

        states = pt.from_numpy(np.vstack(states)).float()
        states_p = pt.from_numpy(np.vstack(states_p)).float()
        actions = pt.from_numpy(np.vstack(actions)).long()
        rewards = pt.from_numpy(np.vstack(rewards)).float()
        terminals = pt.from_numpy(np.vstack(terminals)).float()

        next_q_values = self.policy(states_p).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + 0.99 * next_q_values * (1 - terminals)
        current_q_values = self.policy(states).gather(1, actions)

        loss = self.policy.loss(current_q_values, q_targets)
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        # # Zero the gradients
        # self.policy.optimizer.zero_grad()

        # # Forward pass
        # outputs_pred = self.policy.model(pt.Tensor(np.array(X)))

        # # Compute loss
        # loss = self.policy.loss(outputs_pred, pt.Tensor(Y))

        # # Backward pass
        # loss.backward()

        # # Update weights
        # self.policy.optimizer.step()

    def decay(self):
        self.epsilon = self.epsilon * 0.996
