import numpy as np
from snn.network import Network

class Pattern_Matcher_3(Network):
    def __init__(self, pattern):
        super().__init__(2, 1)
        # Input 0 is ST
        # Input 1 is ~ST
        n = len(pattern)
        self.add_neuron(("AND", n, 0))
        for i in range(n):
            self.add_synapse((f"Input {1 - int(pattern[i])}", "AND", 1, n - i))
        self.add_synapse(("AND", "Output 0", 1))
        self.setup_network()

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            inputs = args[0]
            if isinstance(inputs, str):
                inputs = [int(ch) for ch in inputs]
            args = args[1:]
            output = []
            for i in range(len(inputs)):
                kwargs["inputs"] = np.array([inputs[i], 1 - inputs[i]])
                output.append(self.step(*args, **kwargs).item())
            return np.array(output)

        return self.step(*args, **kwargs)

matcher = Pattern_Matcher_3("1010")
print(matcher([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]))
print(matcher("000100101001110001011011101000"))
