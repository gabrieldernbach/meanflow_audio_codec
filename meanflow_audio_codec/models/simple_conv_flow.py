"""Simple convolutional flow model with AdaLN, similar to MLP but with convs."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from meanflow_audio_codec.utils import sinusoidal_embedding

# ============================================================================
# BUILDING BLOCKS - Normalization and convolution components
# ============================================================================


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with conditioning."""
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, H, W, C] or [B, ..., C]
            condition: Conditioning tensor of shape [B, cond_dim]
        Returns:
            Normalized and modulated tensor of same shape as x
        """
        # LayerNorm (no learnable scale/bias, provided by condition)
        x = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)

        # Generate scale and shift from condition
        scale_shift = nn.Dense(2 * self.features)(condition)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)

        # Expand to match spatial dimensions: [B, C] -> [B, 1, 1, C]
        while scale.ndim < x.ndim:
            scale = jnp.expand_dims(scale, axis=1)
            shift = jnp.expand_dims(shift, axis=1)

        # Apply adaptive normalization: (1 + scale) * x + shift
        return (1.0 + scale) * x + shift


# ============================================================================
# CONVOLUTION BLOCKS - Simple 3x3 conv with AdaLN
# ============================================================================


class SimpleConvBlock(nn.Module):
    """Simple 3x3 conv block with AdaLN."""
    features: int
    condition_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, H, W, C]
            condition: Conditioning tensor of shape [B, condition_dim]
        Returns:
            Output tensor of shape [B, H, W, features]
        """
        # 3x3 convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
        )(x)

        # AdaLN
        x = AdaLN(features=self.features)(x, condition)

        # GELU activation
        x = jax.nn.gelu(x, approximate=True)

        return x


# ============================================================================
# MAIN MODEL - Simple convolutional flow with downsampling/upsampling
# ============================================================================


class SimpleConvFlow(nn.Module):
    """
    Simple convolutional flow model with downsampling and upsampling.
    
    Architecture: Down (pool) -> Conv blocks -> Up (upsample) -> Conv blocks
    Similar to MLP but with convolutions.
    """
    noise_dimension: int
    condition_dimension: int
    latent_dimension: int
    image_size: int = 28
    base_channels: int = 64
    num_latent_tokens: int = 32

    def setup(self):
        # Infer spatial size
        self.spatial_size = int(math.sqrt(self.noise_dimension))
        
        # Project latents to condition dimension
        self.latent_proj = nn.Dense(self.condition_dimension)

        # Input projection with bottleneck
        bottleneck_dim = 256
        self.input_proj1 = nn.Dense(bottleneck_dim)
        self.input_proj2 = nn.Dense(
            self.spatial_size * self.spatial_size * self.base_channels
        )

        # Downsampling path: 28x28 -> 14x14 -> 7x7
        self.down_conv1 = SimpleConvBlock(
            features=self.base_channels, condition_dim=self.condition_dimension
        )
        self.down_conv2 = SimpleConvBlock(
            features=self.base_channels * 2, condition_dim=self.condition_dimension
        )

        # Middle block at 7x7
        self.mid_conv = SimpleConvBlock(
            features=self.base_channels * 2, condition_dim=self.condition_dimension
        )

        # Channel reduction before upsampling
        self.channel_reduce = nn.Conv(
            features=self.base_channels,
            kernel_size=(1, 1),
            use_bias=True,
        )

        # Upsampling path: 7x7 -> 14x14 -> 28x28
        self.up_conv1 = SimpleConvBlock(
            features=self.base_channels, condition_dim=self.condition_dimension
        )
        self.up_conv2 = SimpleConvBlock(
            features=self.base_channels, condition_dim=self.condition_dimension
        )

        # Output projection with bottleneck
        bottleneck_dim = 256
        self.output_proj1 = nn.Dense(bottleneck_dim)
        self.output_proj2 = nn.Dense(self.noise_dimension)

    def __call__(
        self, x: jnp.ndarray, time: jnp.ndarray, latents: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, noise_dimension]
            time: Time tensor of shape [B, 2] (t and h)
            latents: Optional latent tokens of shape [B, num_latent_tokens, latent_dim].
                     If None, unconditional mode (for classifier-free guidance).
        Returns:
            Output tensor of shape [B, noise_dimension]
        """
        # Time embeddings: [B, 2] -> [B, condition_dim] each
        t_emb = sinusoidal_embedding(time[:, 0], self.condition_dimension)
        h_emb = sinusoidal_embedding(time[:, 1], self.condition_dimension)
        condition = t_emb + h_emb  # [B, condition_dim]
        
        # Add latent conditioning if provided
        if latents is not None:
            # Flatten latents: [B, num_latent_tokens, latent_dim] -> [B, num_latent_tokens * latent_dim]
            latents_flat = latents.reshape(latents.shape[0], -1)
            # Project to condition_dimension
            latent_cond = self.latent_proj(latents_flat)
            condition = condition + latent_cond

        # Project input to spatial representation
        # x: [B, noise_dim] -> [B, bottleneck] -> [B, H*W*C]
        x = self.input_proj1(x)
        x = jax.nn.gelu(x, approximate=True)
        x_spatial = self.input_proj2(x)
        # Reshape: [B, H*W*C] -> [B, H, W, C]
        x_spatial = x_spatial.reshape(
            x_spatial.shape[0], self.spatial_size, self.spatial_size, self.base_channels
        )

        # Downsampling path: 28x28 -> 14x14 -> 7x7
        x = self.down_conv1(x_spatial, condition)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = self.down_conv2(x, condition)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Middle block
        x = self.mid_conv(x, condition)

        # Reduce channels before upsampling
        x = self.channel_reduce(x)

        # Upsampling path: 7x7 -> 14x14 -> 28x28
        x = jax.image.resize(
            x,
            (x.shape[0], self.spatial_size // 2, self.spatial_size // 2, x.shape[-1]),
            method="nearest",
        )
        x = self.up_conv1(x, condition)

        x = jax.image.resize(
            x,
            (x.shape[0], self.spatial_size, self.spatial_size, x.shape[-1]),
            method="nearest",
        )
        x = self.up_conv2(x, condition)

        # Flatten and project to output
        x_flat = x.reshape(x.shape[0], -1)
        x = self.output_proj1(x_flat)
        x = jax.nn.gelu(x, approximate=True)
        output = self.output_proj2(x)

        return output

