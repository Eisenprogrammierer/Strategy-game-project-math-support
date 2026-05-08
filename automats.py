from transitions import Machine


class MelleyMachine:
    def __init__(self, name: str, states: list, inputs: dict, outputs: dict) -> None:
        self.name = name
        self.states = states
        self.inputs = inputs
        self.outputs = outputs

        self.machine = Machine(model=self, states=self.states, initial=self.states[0])

    def build(self, triggers, sources, dests):
        for trigger in triggers:
            self.machine.add_transition(trigger=trigger, source=sources[trigger], dest=dests[trigger])
