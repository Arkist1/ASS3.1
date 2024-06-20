import dataclasses


@dataclasses.dataclass
class Transition:
    state: tuple
    next_state: tuple
    action: int
    reward: float

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)
