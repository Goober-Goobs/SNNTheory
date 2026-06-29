import numpy as np

class Network:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.threshold = None
        self.decay = None
        self.voltage = None
        self.synapse_mat = None
        self.x = None

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_synapse(self, synapse):
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

    def step(self):
        self.voltage *= self.decay
        self.voltage += self.x[-1] @ self.synapse_mat
        fires = (self.voltage >= self.threshold)
        self.voltage[fires] = 0
        self.x = np.concatenate([self.x, fires], axis=0)
