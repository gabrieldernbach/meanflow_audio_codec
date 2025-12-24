"""Preprocessing pipeline composition utilities.

Provides utilities for composing multiple preprocessing steps into pipelines.
"""

from typing import Callable

import jax.numpy as jnp

from meanflow_audio_codec.preprocessing.tokenization import TokenizationStrategy


class PreprocessingPipeline:
    """Composes multiple preprocessing steps into a pipeline.
    
    Steps are applied sequentially in order.
    """
    
    def __init__(self, steps: list[Callable[[jnp.ndarray], jnp.ndarray]]):
        """Initialize preprocessing pipeline.
        
        Args:
            steps: List of preprocessing functions, each taking and returning jnp.ndarray
        """
        self.steps = steps
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply all preprocessing steps sequentially.
        
        Args:
            x: Input data
        
        Returns:
            Preprocessed data
        """
        for step in self.steps:
            x = step(x)
        return x
    
    def tokenize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Alias for __call__ for consistency with TokenizationStrategy."""
        return self(x)
    
    def detokenize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply inverse of all preprocessing steps in reverse order.
        
        Args:
            tokens: Preprocessed tokens
        
        Returns:
            Reconstructed data
        
        Note:
            This requires steps to be invertible. If a step is not invertible,
            this will raise NotImplementedError.
        """
        # Reverse order and apply inverse
        for step in reversed(self.steps):
            if hasattr(step, "detokenize"):
                tokens = step.detokenize(tokens)
            elif hasattr(step, "__invert__"):
                tokens = step.__invert__(tokens)
            else:
                raise NotImplementedError(
                    f"Step {step} does not support detokenize/inverse operation"
                )
        return tokens


class Compose:
    """Composes multiple tokenization strategies.
    
    Similar to PreprocessingPipeline but specifically for TokenizationStrategy objects.
    """
    
    def __init__(self, strategies: list[TokenizationStrategy]):
        """Initialize composition of tokenization strategies.
        
        Args:
            strategies: List of tokenization strategies to apply sequentially
        """
        self.strategies = strategies
    
    def tokenize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply all tokenization strategies sequentially.
        
        Args:
            x: Input data
        
        Returns:
            Tokenized data
        """
        for strategy in self.strategies:
            x = strategy.tokenize(x)
        return x
    
    def detokenize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply inverse tokenization strategies in reverse order.
        
        Args:
            tokens: Tokenized data
        
        Returns:
            Reconstructed data
        """
        for strategy in reversed(self.strategies):
            tokens = strategy.detokenize(tokens)
        return tokens


def create_mdct_pipeline(
    window_size: int = 512,
    hop_size: int | None = None,
) -> PreprocessingPipeline:
    """Create a preprocessing pipeline with MDCT tokenization.
    
    Args:
        window_size: MDCT window size
        hop_size: MDCT hop size (defaults to window_size // 2)
    
    Returns:
        PreprocessingPipeline with MDCT tokenization
    """
    from meanflow_audio_codec.preprocessing.tokenization import MDCTTokenization
    
    mdct = MDCTTokenization(window_size=window_size, hop_size=hop_size)
    return PreprocessingPipeline([mdct.tokenize])


def create_reshape_pipeline(
    patch_size: int | tuple[int, int] | None = None,
    patch_length: int | None = None,
    image_size: int | tuple[int, int] | None = None,
) -> PreprocessingPipeline:
    """Create a preprocessing pipeline with reshape tokenization.
    
    Args:
        patch_size: Patch size for images
        patch_length: Patch length for audio
        image_size: Image size for images
    
    Returns:
        PreprocessingPipeline with reshape tokenization
    """
    from meanflow_audio_codec.preprocessing.tokenization import ReshapeTokenization
    
    reshape = ReshapeTokenization(
        patch_size=patch_size,
        patch_length=patch_length,
        image_size=image_size,
    )
    return PreprocessingPipeline([reshape.tokenize])

