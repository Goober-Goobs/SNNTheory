import numpy as np

class Network:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [(0, 0) for i in range(num_inputs + num_outputs)]
        self.names = [f"Input {i}" for i in range(num_inputs)] + [f"Output {i}" for i in range(num_outputs)]
        self.synapses = []
        self.threshold = None
        self.decay = None
        self.voltage = None
        self.synapse_mat = None
        self.x = None

    def add_neuron(self, neuron):
        # (name, threshold, decay)
        self.names.append(neuron[0])
        self.neurons.append(neuron[1:])

    def add_synapse(self, synapse):
        for i in range(2):
            if isinstance(synapse[i], str):
                if synapse[i] not in self.names:
                    raise ValueError("Neuron name '" + synapse[i] + "' not found.")
                synapse[i] = self.names.index(synapse[i])
        # (presynaptic, postsynaptic, weight)
        self.synapses.append(synapse)

    def setup_network(self):
        n = len(self.neurons)
        threshold, decay = zip(*self.neurons)
        self.threshold = np.array(threshold)
        self.decay = np.array(decay)
        self.voltage = np.zeros_like(threshold)

        self.synapse_mat = np.zeros((n, n))
        for synapse in self.synapses:
            self.synapse_mat[synapse[0], synapse[1]] += synapse[2]
        self.x = np.zeros((1, n))

    def step(self, inputs):
        self.x[-1][:self.num_inputs] = inputs

        self.voltage *= self.decay
        self.voltage += self.x[-1] @ self.synapse_mat
        fires = (self.voltage >= self.threshold)
        self.voltage[fires] = 0
        self.x = np.concatenate([self.x, fires], axis=0)

        return fires[self.num_inputs:self.num_inputs + self.num_outputs]
