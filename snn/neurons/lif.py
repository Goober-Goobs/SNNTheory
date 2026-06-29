import torch
 
from ..module import Module
from ..parameter import Parameter
 
 
class LIF(Module):
    """
    Membrane:
 
        v[t] = decay * v[t-1] + I[t] + bias
        s[t] = 1  if  v[t] >= threshold  else  0
        v[t] = 0  wherever  s[t] == 1            # hard reset to 0 on a spike
 
    where "I[t]" is the input current for this timestep (already weighted by
    the synapses feeding the neuron).
 
    Parameter conventions:
    decay:
        decay is the fraction of voltage kept, not lost.
        decay = 0 -> memoryless: a pure threshold gate with no integration
          across time. This is exactly what the AND / OR gates use.
        decay = 1 -> perfect integrator (no leak).
        0 < decay < 1" -> leaky integrator.
    threshold : firing threshold "T".
    bias : constant tonic current added every step (default 0). Used to build a
        spiking inverter (see "NOT"), which must fire on the absence of an
        input spike.
 
    State:
    "v" is the membrane potential. It persists across timesteps (that is the
    point of integration) and is cleared by method "reset_state". It is created
    lazily on the first "forward" so it matches the input's shape/device.
 
    Notes
    Synaptic weights live on the *connections* feeding the neuron, matching the
    framework's delay-1 weighted-synapse model, so the neuron integrates the
    incoming current directly. "threshold" / "decay" / "bias" are stored
    as non-learnable constants (buffers) by default; pass "learnable=True" to
    make "threshold" and "decay" optimizable class Parameter.
    """
 
    def __init__(self, threshold=1.0, decay=0.0, bias=0.0, learnable=False):
        super().__init__()
 
        if learnable:
            self.threshold = Parameter(float(threshold))
            self.decay = Parameter(float(decay))
        else:
            # plain tensors -> registered as buffers by Module.__setattr__
            self.threshold = torch.tensor(float(threshold))
            self.decay = torch.tensor(float(decay))
 
        # constant tonic current (buffer)
        self.bias = torch.tensor(float(bias))
 
        # membrane potential: lazily created on first forward (see forward()).
        self.v = None
 
    def reset_state(self):
        super().reset_state()
        self.v = None

    @staticmethod
    def _value(x):
        return x.data if isinstance(x, Parameter) else x
 
    def forward(self, x):
        """Advance one timestep given input current "x"."""

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
 
        # (re)allocate membrane state to match the current input shape/device
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x)
 
        decay = self._value(self.decay)
        thr = self._value(self.threshold)
        bias = self._value(self.bias)
 
        self.v = decay * self.v + x + bias
        spike = (self.v >= thr).float()
        self.v = self.v * (1.0 - spike)      # hard reset to 0 where it fired
        return spike
 
    def extra_repr(self):
        return (f"threshold={float(self._value(self.threshold))}, "
                f"decay={float(self._value(self.decay))}, "
                f"bias={float(self._value(self.bias))}")   
 