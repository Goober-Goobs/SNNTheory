import torch

class Parameter:
    """
    Lightweight learnable parameter wrapper.

    Stores a torch.Tensor in `data` while behaving similarly to a tensor.
    """

    def __init__(self, data):

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        self.data = data

    # -------------------------------------------------
    # Convenience constructors
    # -------------------------------------------------

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(torch.zeros(*shape, **kwargs))

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(torch.ones(*shape, **kwargs))

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(torch.randn(*shape, **kwargs))

    # -------------------------------------------------
    # Device movement
    # -------------------------------------------------

    def to(self, device):
        self.data = self.data.to(device)
        return self

    # -------------------------------------------------
    # Tensor properties
    # -------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    # -------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __repr__(self):

        return (
            f"Parameter(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )
