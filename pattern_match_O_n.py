from snn.network import Network

class OR_Network(Network):
    def __init__(self):
        super().__init__(2, 1)
        self.add_neuron(("OR", 1, 0))
        self.add_synapse(("Input 0", "OR", 1))
        self.add_synapse(("Input 1", "OR", 1))
        self.add_synapse(("OR", "Output 0", 1))
        self.setup_network()

matcher = OR_Network()
print(matcher.step())
print(matcher.step([0, 1]))
print(matcher.step())
print(matcher.step([1, 0]))
print(matcher.step())
print(matcher.step())
