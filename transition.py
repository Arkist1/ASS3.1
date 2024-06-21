import dataclasses


@dataclasses.dataclass
class Transition:
    state: tuple
    next_state: tuple
    action: int  
    reward: float
    terminal: bool

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def serialize(self):
        return {"state" : [str(x) for x in self.state],
                "next_state": [str(x) for x in self.next_state],
                "action": str(self.action),
                "reward": str(self.reward),
                "terminal": str(self.terminal)}