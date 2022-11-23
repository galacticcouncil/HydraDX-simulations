class Oracle:
    def __init__(self, decay_factor: float = 0, sma_equivalent_length: int = 0):
        if decay_factor:
            self.decay_factor = decay_factor
        elif sma_equivalent_length:
            self.decay_factor = 2 / (sma_equivalent_length + 1)
        else:
            self.decay_factor = 0.5
        self.values = {}
        self.current = {}

    def add(self, attribute: str, value: float = None):
        if attribute not in self.current:
            self.current[attribute] = value
        else:
            self.current[attribute] += value

    def update(self, attribute: str, value: float = None):
        if value is None:
            if attribute in self.current:
                value = self.current[attribute]
                self.current[attribute] = 0
            else:
                value = 0
        if attribute not in self.values:
            self.values[attribute] = value
        else:
            self.values[attribute] = (1 - self.decay_factor) * self.values[attribute] + self.decay_factor * value

    def get(self, attribute: str):
        return self.values[attribute]
