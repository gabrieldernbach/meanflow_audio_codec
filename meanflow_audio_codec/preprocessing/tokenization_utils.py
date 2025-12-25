"""Utilities for creating and using tokenization strategies from configuration."""

import jax.numpy as jnp
import numpy as np

from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.preprocessing.mdct import MDCTConfig
from meanflow_audio_codec.preprocessing.tokenization import (
    MDCTTokenization,
    ReshapeTokenization,
    TokenizationStrategy,
)


def create_tokenization_strategy(config: TrainFlowConfig) -> TokenizationStrategy | None:
    """Create tokenization strategy from configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Tokenization strategy instance, or None if no tokenization should be applied
    """
    tokenization_strategy = config.tokenization_strategy
    
    if tokenization_strategy is None:
        return None
    
    tokenization_config = config.tokenization_config or {}
    
    if tokenization_strategy == "mdct":
        window_size = tokenization_config.get("window_size", 512)
        hop_size = tokenization_config.get("hop_size")
        mdct_config = MDCTConfig(window_size=window_size, hop_size=hop_size)
        return MDCTTokenization(config=mdct_config)
    
    elif tokenization_strategy == "reshape":
        patch_size = tokenization_config.get("patch_size")
        patch_length = tokenization_config.get("patch_length")
        image_size = tokenization_config.get("image_size")
        
        # Convert image_size to tuple if it's a list
        if image_size is not None and isinstance(image_size, list):
            image_size = tuple(image_size)
        
        # Convert patch_size to tuple if it's a list
        if patch_size is not None and isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        
        return ReshapeTokenization(
            patch_size=patch_size,
            patch_length=patch_length,
            image_size=image_size,
        )
    
    else:
        raise ValueError(
            f"Unknown tokenization_strategy: {tokenization_strategy}. "
            "Must be one of: 'mdct', 'reshape'"
        )


def compute_tokenized_dimension(
    tokenization: TokenizationStrategy,
    original_dimension: int,
    dataset: str,
) -> int:
    """Compute the flattened dimension after tokenization.
    
    This creates a dummy input with the original dimension, tokenizes it,
    and returns the flattened size.
    
    Args:
        tokenization: Tokenization strategy
        original_dimension: Original data dimension (e.g., 784 for MNIST)
        dataset: Dataset name ('mnist' or 'audio')
        
    Returns:
        Flattened dimension after tokenization (n_tokens * token_dim)
    """
    # Create dummy input based on dataset
    batch_size = 1
    
    if dataset == "mnist":
        # MNIST: [B, 784] (flattened 28x28)
        dummy_input = jnp.zeros((batch_size, original_dimension), dtype=jnp.float32)
    elif dataset == "audio":
        # Audio: [B, T] where T = original_dimension
        dummy_input = jnp.zeros((batch_size, original_dimension), dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Tokenize
    tokens = tokenization.tokenize(dummy_input)
    
    # Flatten: [B, n_tokens, token_dim] -> [B, n_tokens * token_dim]
    flattened_dim = tokens.shape[1] * tokens.shape[2]
    
    return int(flattened_dim)


def compute_token_shape(
    tokenization: TokenizationStrategy,
    original_dimension: int,
    dataset: str,
) -> tuple[int, int]:
    """Compute the token shape (n_tokens, token_dim) after tokenization.
    
    This creates a dummy input with the original dimension, tokenizes it,
    and returns the token shape.
    
    Args:
        tokenization: Tokenization strategy
        original_dimension: Original data dimension (e.g., 784 for MNIST)
        dataset: Dataset name ('mnist' or 'audio')
        
    Returns:
        Tuple of (n_tokens, token_dim)
    """
    # Create dummy input based on dataset
    batch_size = 1
    
    if dataset == "mnist":
        # MNIST: [B, 784] (flattened 28x28)
        dummy_input = jnp.zeros((batch_size, original_dimension), dtype=jnp.float32)
    elif dataset == "audio":
        # Audio: [B, T] where T = original_dimension
        dummy_input = jnp.zeros((batch_size, original_dimension), dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Tokenize
    tokens = tokenization.tokenize(dummy_input)
    
    return (int(tokens.shape[1]), int(tokens.shape[2]))

