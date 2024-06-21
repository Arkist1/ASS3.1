import random
import numpy as np

from memory import Memory
from policy import Policy
from transition import Transition

import torch as pt

import json
import os


class Agent:
    def __init__(self, memory_size, memory_path, sample_size, epsilon, discount, lr, policy, dq_policy, decay_amt, averaging_rate, doubleq) -> None:
        self.memory = Memory(memory_size)
        self.policy = Policy(policy, dq_policy, doubleq, lr=lr)

        self.load_memory(memory_path)

        self.moves = [0, 1, 2, 3]

        self.epsilon = epsilon
        self.discount = discount
        self.sample_size = sample_size
        self.decay_amt = decay_amt
        self.averaging_rate = averaging_rate
        
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
            if not len(self.memory.transition_deque) >= self.sample_size:
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

        # Compute loss
        loss = self.policy.loss(pt.stack(X), pt.Tensor(Y))

        # Zero the gradients
        self.policy.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.policy.optimizer.step()

    def save_memory(self, path):
        with open(path, "w") as outfile:
            for x in self.memory.transition_deque:
                outfile.write(json.dumps(x.serialize()) + "\n")

    def load_memory(self, path):
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            for line in f.readlines():
                vars = json.loads(line)
                tr = Transition(state=[float(x) for x in vars["state"]], 
                                             next_state=[float(x) for x in vars["next_state"]], 
                                             action=int(vars["action"]), 
                                             reward=float(vars["reward"]), 
                                             terminal=bool(["terminal"]))
                
                self.memory.store(tr)

    def double_train(self):
        if not self.memory_filled:
            if not len(self.memory.transition_deque) >= self.sample_size:
                return
            self.memory_filled = True

        samples = self.memory.sample(self.sample_size)

        X = []
        Y = []

        for state, state_prime, action, reward, terminal in samples:
            q_prime = list(self.policy.target_model(pt.Tensor(state_prime)))
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + (self.discount * q_prime[a_prime]) * (1 - terminal)

            q_state = self.policy.model(pt.Tensor(state))
            X.append(q_state)

            q_state = list(q_state)
            q_state[action] = a_value

            Y.append(q_state)

        # Compute loss
        loss = self.policy.loss(pt.stack(X), pt.Tensor(Y))

        # Zero the gradients
        self.policy.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.policy.optimizer.step()

        # Update target network params
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.policy.target_model.parameters(), self.policy.model.parameters()):
            target_param.data.copy_(self.averaging_rate * local_param.data + (1.0 - self.averaging_rate) * target_param.data)


    def decay(self):
        self.epsilon = self.epsilon * self.decay_amt
