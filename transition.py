import dataclasses


@dataclasses.dataclass
class Transition:
    state: tuple
    next_state: tuple
    reward: float
    action: int
    
    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)
