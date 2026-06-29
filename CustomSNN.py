import numpy as np

class Network:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.threshold = None
        self.decay = None
        self.voltage = None
        self.synapse_mat = None
        self.bias = None
        self.x = None

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_synapse(self, synapse):
        # (presynaptic, postsynaptic, weight)
        self.synapses.append(synapse)

    def setup_network(self):
        threshold = [neuron[0] for neuron in self.neurons]
        decay = [neuron[1] for neuron in self.neurons]
        bias = [neuron[2] if len(neuron) > 2 else 0.0 for neuron in self.neurons]
  
        self.threshold = np.array(threshold, dtype=float)
        self.decay = np.array(decay, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.voltage = np.zeros(n, dtype=float)
        

        self.synapse_mat = np.zeros((n, n))
        for synapse in self.synapses:
            self.synapse_mat[synapse[0], synapse[1]] += synapse[2]
        self.x = np.zeros((1, n))

    def step(self, ext=None):
        self.voltage *= self.decay
        self.voltage += self.x[-1] @ self.synapse_mat
        self.voltage += self.bias
        if ext is not None:
            self.voltage += ext
        fires = self.voltage >= self.threshold
        self.voltage[fires] = 0.0
        self.x = np.concatenate([self.x, fires[None, :].astype(float)], axis=0)
        return fires
  