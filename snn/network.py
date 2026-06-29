import numpy as np

class Network:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [(1, 0) for _ in range(num_inputs)] + [(1, 0) for _ in range(num_outputs)]
        self.names = [f"Input {i}" for i in range(num_inputs)] + [f"Output {i}" for i in range(num_outputs)]
        self.synapses = []
        self.threshold = None
        self.decay = None
        self.index = None
        self.voltage = None
        self.pre = None
        self.post = None
        self.weights = None
        self.delays = None
        self.x = None

    def add_neuron(self, neuron):
        # (name, threshold, decay)
        self.names.append(neuron[0])
        self.neurons.append(neuron[1:])

    def add_synapse(self, synapse):
        # (presynaptic, postsynaptic, weight, delay=1)
        synapse = list(synapse)
        for i in range(2):
            if isinstance(synapse[i], str):
                if synapse[i] not in self.names:
                    raise ValueError("Neuron name '" + synapse[i] + "' not found.")
                synapse[i] = self.names.index(synapse[i])
        # (presynaptic, postsynaptic, weight, delay=1)
        if len(synapse) == 3:
            synapse.append(1)
        self.synapses.append(tuple(synapse))

    def setup_network(self):
        n = len(self.neurons)
        threshold, decay = zip(*self.neurons)
        self.threshold = np.array(threshold).astype(np.float64)
        self.decay = np.array(decay).astype(np.float64)
        self.voltage = np.zeros_like(threshold).astype(np.float64)

        pre, post, weights, delays = zip(*self.synapses)
        self.pre = np.array(pre)
        self.post = np.array(post)
        self.weights = np.array(weights)
        self.delays = np.array(delays)
        self.x = np.zeros((1, n)).astype(bool)

    def step(self, inputs=None):
        n = self.voltage.shape[0]
        if inputs is not None:
            self.x[-1][:self.num_inputs] = np.array(inputs)

        self.voltage *= self.decay
        pos_delay = np.maximum(self.x.shape[0] - self.delays, 0)

        np.add.at(self.voltage, self.post, self.x[pos_delay, self.pre] * self.weights)
        fires = (self.voltage >= self.threshold)
        self.voltage[fires] = 0
        self.x = np.concatenate([self.x, fires[np.newaxis, :]], axis=0)

        return fires[self.num_inputs:self.num_inputs + self.num_outputs]

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            inputs = args[0]
            args = args[1:]
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            kwargs["inputs"] = inputs
            if isinstance(inputs, np.ndarray):
                if inputs.ndim > 1:
                    outputs = []
                    for i in range(inputs.shape[0]):
                        kwargs["inputs"] = inputs[i]
                        outputs.append(self.step(*args, **kwargs))
                        # outputs.append((*inputs[i].tolist(), np.inf, *self.step(*args, **kwargs).tolist()))
                    # return np.array(outputs)
                    return inputs, np.array(outputs)

        return self.step(*args, **kwargs)
