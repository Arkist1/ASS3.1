from dataclasses import dataclass


@dataclass
class Transition:
    state: tuple
    next_state: tuple
    reward: float
    action: int
