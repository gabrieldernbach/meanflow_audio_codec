"""Tokenization strategies for converting data to tokens.

This module provides different tokenization strategies:
- MDCT: Modified Discrete Cosine Transform (frequency domain)
- Reshape: Patch-based tokenization (spatial domain)
"""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np
from einops import rearrange

from meanflow_audio_codec.preprocessing.mdct import MDCTConfig, mdct, imdct


class TokenizationStrategy(ABC):
    """Abstract base class for tokenization strategies."""
    
    @abstractmethod
    def tokenize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert input to tokens.
        
        Args:
            x: Input data, shape depends on data type
        
        Returns:
            Tokens, shape [B, n_tokens, token_dim]
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Convert tokens back to original format.
        
        Args:
            tokens: Tokens, shape [B, n_tokens, token_dim]
        
        Returns:
            Reconstructed data, same shape as original input
        """
        pass


class MDCTTokenization(TokenizationStrategy):
    """MDCT-based tokenization for audio signals.
    
    Converts time-domain audio to frequency-domain MDCT coefficients.
    """
    
    def __init__(
        self,
        window_size: int = 512,
        hop_size: int | None = None,
        config: MDCTConfig | None = None,
    ):
        """Initialize MDCT tokenization.
        
        Args:
            window_size: Size of MDCT window
            hop_size: Hop size between windows (defaults to window_size // 2)
            config: Optional MDCTConfig object (overrides window_size/hop_size)
        """
        if config is not None:
            self.config = config
        else:
            self.config = MDCTConfig(
                window_size=window_size,
                hop_size=hop_size,
            )
    
    def tokenize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert audio to MDCT tokens.
        
        Args:
            x: Audio signal, shape [B, T] (mono) or [B, T, C] (multi-channel)
        
        Returns:
            MDCT coefficients, shape [B, n_frames, window_size] or [B, n_frames, window_size*C]
        """
        if x.ndim == 2:
            # Mono: [B, T] -> [B, n_frames, window_size]
            return mdct(x, config=self.config)
        elif x.ndim == 3:
            # Multi-channel: process each channel and concatenate
            # [B, T, C] -> [B, n_frames, window_size*C]
            channels = []
            for c in range(x.shape[2]):
                channel_mdct = mdct(x[:, :, c], config=self.config)
                channels.append(channel_mdct)
            return jnp.concatenate(channels, axis=-1)
        else:
            raise ValueError(f"Invalid input shape for MDCT: {x.shape}")
    
    def detokenize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Convert MDCT tokens back to audio.
        
        Args:
            tokens: MDCT coefficients, shape [B, n_frames, window_size] or [B, n_frames, window_size*C]
        
        Returns:
            Audio signal, shape [B, T] or [B, T, C]
        """
        if tokens.ndim != 3:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}, expected [B, n_frames, ...]")
        
        # Check if multi-channel (window_size*C)
        window_size = self.config.window_size
        if tokens.shape[2] == window_size:
            # Mono: [B, n_frames, window_size] -> [B, T]
            return imdct(tokens, config=self.config)
        elif tokens.shape[2] % window_size == 0:
            # Multi-channel: split and process each channel
            n_channels = tokens.shape[2] // window_size
            channels = []
            for c in range(n_channels):
                start_idx = c * window_size
                end_idx = (c + 1) * window_size
                channel_tokens = tokens[:, :, start_idx:end_idx]
                channel_audio = imdct(channel_tokens, config=self.config)
                channels.append(channel_audio)
            # Stack: [B, T, C]
            return jnp.stack(channels, axis=-1)
        else:
            raise ValueError(
                f"Invalid tokens shape: {tokens.shape}, "
                f"token_dim ({tokens.shape[2]}) must be multiple of window_size ({window_size})"
            )


class ReshapeTokenization(TokenizationStrategy):
    """Reshape-based tokenization using patch extraction.
    
    For images: extracts patches similar to Vision Transformers.
    For audio: reshapes time-domain into patches.
    """
    
    def __init__(
        self,
        patch_size: int | tuple[int, int] | None = None,
        patch_length: int | None = None,
        image_size: int | tuple[int, int] | None = None,
    ):
        """Initialize reshape tokenization.
        
        Args:
            patch_size: Patch size for images, either int (square) or (H, W) tuple.
                        For MNIST (28x28), common values are 4 or 7.
            patch_length: Patch length for audio (number of samples per patch).
                         For audio, this is the token dimension.
            image_size: Image size for images, either int (square) or (H, W) tuple.
                        If None, inferred from input shape.
        """
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.image_size = image_size
    
    def tokenize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert input to patch tokens.
        
        Args:
            x: Input data:
                - Images: [B, H, W] or [B, H*W] (flattened)
                - Audio: [B, T] or [B, T, C]
        
        Returns:
            Tokens, shape [B, n_patches, patch_dim]
        """
        if x.ndim == 2:
            # Could be flattened image or audio
            # Try to infer: if shape suggests image (e.g., 784 = 28*28), treat as image
            # Otherwise treat as audio
            if self.image_size is not None or self.patch_size is not None:
                # Image case
                return self._tokenize_image(x)
            elif self.patch_length is not None:
                # Audio case
                return self._tokenize_audio(x)
            else:
                # Default: try to infer from shape
                if x.shape[1] in (784, 28 * 28):  # MNIST-like
                    return self._tokenize_image(x)
                else:
                    return self._tokenize_audio(x)
        elif x.ndim == 3:
            if x.shape[2] in (1, 3):  # Image with channel dimension
                return self._tokenize_image(x)
            else:  # Audio with multiple channels
                return self._tokenize_audio(x)
        else:
            raise ValueError(f"Invalid input shape for reshape tokenization: {x.shape}")
    
    def _tokenize_image(self, x: jnp.ndarray) -> jnp.ndarray:
        """Tokenize image into patches."""
        if x.ndim == 2:
            # Flattened: [B, H*W] -> [B, H, W]
            if self.image_size is None:
                # Infer from shape (assume square)
                total_pixels = x.shape[1]
                h = w = int(np.sqrt(total_pixels))
            else:
                h, w = (
                    (self.image_size, self.image_size)
                    if isinstance(self.image_size, int)
                    else self.image_size
                )
            x = x.reshape(x.shape[0], h, w)
        
        # Now x is [B, H, W] or [B, H, W, C]
        if x.ndim == 3:
            # Add channel dimension: [B, H, W] -> [B, H, W, 1]
            x = x[..., None]
        
        # Determine patch size
        if self.patch_size is None:
            # Default: use 4x4 patches for 28x28 images
            patch_h = patch_w = 4
        elif isinstance(self.patch_size, int):
            patch_h = patch_w = self.patch_size
        else:
            patch_h, patch_w = self.patch_size
        
        # Extract patches: [B, H, W, C] -> [B, n_patches, patch_h*patch_w*C]
        # Using einops rearrange
        tokens = rearrange(
            x,
            "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
            p1=patch_h,
            p2=patch_w,
        )
        return tokens
    
    def _tokenize_audio(self, x: jnp.ndarray) -> jnp.ndarray:
        """Tokenize audio into patches."""
        if x.ndim == 3:
            # Multi-channel: [B, T, C] -> flatten channels
            # [B, T, C] -> [B, T*C]
            x = x.reshape(x.shape[0], -1)
        
        # Now x is [B, T]
        if self.patch_length is None:
            # Default: 128 samples per patch
            patch_length = 128
        else:
            patch_length = self.patch_length
        
        # Extract patches: [B, T] -> [B, n_patches, patch_length]
        # Pad if necessary
        T = x.shape[1]
        n_patches = (T + patch_length - 1) // patch_length
        padded_length = n_patches * patch_length
        
        if T < padded_length:
            # Pad with zeros
            padding = jnp.zeros((x.shape[0], padded_length - T), dtype=x.dtype)
            x = jnp.concatenate([x, padding], axis=1)
        
        # Reshape: [B, T] -> [B, n_patches, patch_length]
        tokens = x.reshape(x.shape[0], n_patches, patch_length)
        return tokens
    
    def detokenize(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Convert patch tokens back to original format.
        
        Args:
            tokens: Tokens, shape [B, n_patches, patch_dim]
        
        Returns:
            Reconstructed data
        """
        # For images: [B, n_patches, patch_h*patch_w*C] -> [B, H, W, C] -> [B, H, W]
        # For audio: [B, n_patches, patch_length] -> [B, T]
        
        # Try to infer from patch_dim
        patch_dim = tokens.shape[2]
        
        # Check if it's likely an image patch (square number or multiple of small squares)
        if self.patch_size is not None or self.image_size is not None:
            return self._detokenize_image(tokens)
        elif self.patch_length is not None:
            return self._detokenize_audio(tokens)
        else:
            # Heuristic: if patch_dim is a perfect square or small, likely image
            # Otherwise likely audio
            sqrt_dim = int(np.sqrt(patch_dim))
            if sqrt_dim * sqrt_dim == patch_dim and sqrt_dim <= 16:
                return self._detokenize_image(tokens)
            else:
                return self._detokenize_audio(tokens)
    
    def _detokenize_image(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Detokenize image patches."""
        B, n_patches, patch_dim = tokens.shape
        
        # Infer patch size
        if self.patch_size is None:
            # Try to infer from patch_dim
            sqrt_dim = int(np.sqrt(patch_dim))
            if sqrt_dim * sqrt_dim == patch_dim:
                patch_h = patch_w = sqrt_dim
                n_channels = 1
            else:
                # Try common sizes
                for p in [2, 4, 7, 8]:
                    if patch_dim % (p * p) == 0:
                        patch_h = patch_w = p
                        n_channels = patch_dim // (p * p)
                        break
                else:
                    # Default
                    patch_h = patch_w = 4
                    n_channels = 1
        elif isinstance(self.patch_size, int):
            patch_h = patch_w = self.patch_size
            n_channels = patch_dim // (self.patch_size ** 2)
        else:
            patch_h, patch_w = self.patch_size
            n_channels = patch_dim // (patch_h * patch_w)
        
        # Infer image size
        if self.image_size is None:
            # Infer from number of patches (assume square)
            n_patches_per_side = int(np.sqrt(n_patches))
            h = w = n_patches_per_side * patch_h
        else:
            h, w = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else self.image_size
            )
            n_patches_per_side_h = h // patch_h
            n_patches_per_side_w = w // patch_w
        
        # Reconstruct: [B, n_patches, patch_h*patch_w*C] -> [B, H, W, C]
        x = rearrange(
            tokens,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=n_patches_per_side_h,
            w=n_patches_per_side_w,
            p1=patch_h,
            p2=patch_w,
        )
        
        # Remove channel dimension if single channel: [B, H, W, 1] -> [B, H, W]
        if x.shape[3] == 1:
            x = x[..., 0]
        
        return x
    
    def _detokenize_audio(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Detokenize audio patches."""
        # [B, n_patches, patch_length] -> [B, T]
        B, n_patches, patch_length = tokens.shape
        T = n_patches * patch_length
        x = tokens.reshape(B, T)
        return x

