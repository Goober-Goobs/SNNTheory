from snn.network import Network

class DelayTester(Network):
    def __init__(self):
        super().__init__(1, 3)
        self.add_neuron(("AND", 2, 0))
        self.add_neuron(("D1", 1, 0))
        self.add_neuron(("D2", 1, 0))
        self.add_synapse(("Input 0", "D1", 1, 1))
        self.add_synapse(("Input 0", "D2", 1, 2))
        self.add_synapse(("D1", "AND", 1, 1))
        self.add_synapse(("D2", "AND", 1, 1))
        self.add_synapse(("AND", "Output 0", 1, 1))
        self.add_synapse(("D1", "Output 1", 1, 2))
        self.add_synapse(("D2", "Output 2", 1, 2))
        self.setup_network()

double_one = DelayTester()
print(double_one([[0], [0], [1], [1], [0], [1], [1], [0], [0], [0]])[1])
