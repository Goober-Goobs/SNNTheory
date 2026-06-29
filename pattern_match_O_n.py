from snn.network import Network

class PatternMatcher(Network):
    def __init__(self):
        super().__init__(2, 1)
        self.add_neuron(("OR", 0, 0))
