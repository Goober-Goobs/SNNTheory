from collections import OrderedDict
import torch
from .parameter import Parameter


class Module:
    def __init__(self):
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())

    # =====================================================
    # Attribute registration
    # =====================================================

    def __setattr__(self, name, value):

        if name in ("_modules", "_parameters", "_buffers"):
            object.__setattr__(self, name, value)
            return

        # Remove stale registrations
        self._modules.pop(name, None)
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)

        if isinstance(value, Module):
            self.add_module(name, value)

        elif isinstance(value, Parameter):
            self.add_parameter(name, value)

        elif isinstance(value, torch.Tensor):
            self.register_buffer(name, value)

        object.__setattr__(self, name, value)

    # =====================================================

    def add_module(self, name, module):

        self._modules[name] = module

    def add_parameter(self, name, parameter):

        self._parameters[name] = parameter

    def register_buffer(self, name, tensor):

        self._buffers[name] = tensor

    # =====================================================

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() not implemented."
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # =====================================================
    # DFS traversal
    # =====================================================

    def modules(self):

        yield self

        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix=""):

        yield prefix, self

        for name, module in self._modules.items():

            child_prefix = (
                name if prefix == "" else prefix + "." + name
            )

            yield from module.named_modules(child_prefix)

    # =====================================================

    def parameters(self):

        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix=""):

        for name, parameter in self._parameters.items():

            key = name if prefix == "" else prefix + "." + name

            yield key, parameter

        for child_name, module in self._modules.items():

            child_prefix = (
                child_name if prefix == ""
                else prefix + "." + child_name
            )

            yield from module.named_parameters(child_prefix)

    # =====================================================

    def buffers(self):

        for _, b in self.named_buffers():
            yield b

    def named_buffers(self, prefix=""):

        for name, buffer in self._buffers.items():

            key = name if prefix == "" else prefix + "." + name

            yield key, buffer

        for child_name, module in self._modules.items():

            child_prefix = (
                child_name if prefix == ""
                else prefix + "." + child_name
            )

            yield from module.named_buffers(child_prefix)

    # =====================================================
    # Device movement
    # =====================================================

    def to(self, device):

        for parameter in self._parameters.values():
            parameter.to(device)

        for name in self._buffers:
            tensor = self._buffers[name].to(device)
            self._buffers[name] = tensor
            object.__setattr__(self, name, tensor)

        for module in self._modules.values():
            module.to(device)

        return self

    # =====================================================
    # Serialization
    # =====================================================

    def state_dict(self):

        state = OrderedDict()

        for name, parameter in self.named_parameters():
            state[name] = parameter.data.clone()

        for name, buffer in self.named_buffers():
            state[name] = buffer.clone()

        return state

    def load_state_dict(self, state):

        for name, parameter in self.named_parameters():

            if name in state:
                parameter.data = state[name].clone()

        for name, _ in self.named_buffers():

            if name in state:

                parts = name.split(".")
                obj = self

                for p in parts[:-1]:
                    obj = obj._modules[p]

                obj._buffers[parts[-1]] = state[name].clone()
                object.__setattr__(
                    obj,
                    parts[-1],
                    state[name].clone()
                )

    # =====================================================
    # Reset state
    # =====================================================

    def reset_state(self):

        for module in self._modules.values():
            module.reset_state()

    # =====================================================
    # Pretty printing
    # =====================================================

    def _repr(self, depth):

        indent = "  " * depth

        s = f"{indent}{self.__class__.__name__}(\n"

        for name, module in self._modules.items():

            s += f"{indent}  ({name}):\n"
            s += module._repr(depth + 2)

        s += f"{indent})\n"

        return s

    def __repr__(self):
        return self._repr(0)
