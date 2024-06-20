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
            return self.policy.select_action(state)

    def store_transition(self, transition):
        self.memory.store(transition)

    def train(self):
        samples = self.memory.sample(16)

        for state, state_prime, reward, action in samples:
            q_prime = self.policy.forward(state_prime)
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + 0.9 * q_prime[a_prime]
            
            q_state = self.policy.forward(state)
            q_state[a_prime] = a_value




        s_prime = self.select_action()

    def decay(self):
        self.epsilon = self.epsilon / 1.5
