"""Meanflow Audio Codec - MDCT-based autoencoder using Improved Mean Flows.

This package provides implementations of flow-based audio codecs operating
in the MDCT domain, with support for various flow matching methods.
"""

__version__ = "0.1.0"

# Main package exports
from meanflow_audio_codec.models import (
    ConditionalFlow,
    ConditionalResidualBlock,
    MLP,
    TrainState,
)

__all__ = [
    "__version__",
    "ConditionalFlow",
    "ConditionalResidualBlock",
    "MLP",
    "TrainState",
]


