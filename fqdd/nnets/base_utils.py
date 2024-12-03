import torch
from torch.nn import LayerNorm, BatchNorm1d
from fqdd.nnets.normalization import RMSNorm


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


FQDD_ACTIVATIONS = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}


FQDD_NORMALIZES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}
