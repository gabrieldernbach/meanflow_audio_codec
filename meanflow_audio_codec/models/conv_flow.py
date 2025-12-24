import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from meanflow_audio_codec.utils import sinusoidal_embedding

# ============================================================================
# BUILDING BLOCKS - Normalization and convolution components
# ============================================================================


class GlobalResponseNormalization(nn.Module):
    """
    Global Response Normalization (GRN) layer from ConvNeXt V2.
    
    Enhances inter-channel feature competition via L2 normalization.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, H, W, C] or [B, ..., C]
        Returns:
            Normalized tensor of same shape
        """
        # Compute global response: L2 norm across spatial dimensions
        # x: [B, H, W, C] -> [B, 1, 1, C]
        spatial_dims = tuple(range(1, x.ndim - 1))
        gx = jnp.sqrt(jnp.sum(x**2, axis=spatial_dims, keepdims=True))
        
        # Normalize: gx / (mean(gx) + epsilon)
        n = jnp.mean(gx, axis=-1, keepdims=True)
        gx = gx / (n + self.epsilon)
        
        # Apply learnable scaling and bias
        num_channels = x.shape[-1]
        gamma = self.param('gamma', nn.initializers.zeros, (num_channels,))
        beta = self.param('beta', nn.initializers.zeros, (num_channels,))
        
        # Broadcasting: gamma, beta [C] -> [B, H, W, C]
        return x * (gamma + gx) + beta


# ============================================================================
# CONVOLUTION BLOCKS - ConvNeXt-style processing
# ============================================================================


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with small 3x3 convolution.
    
    Architecture: 3x3 Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv
    """
    dim: int
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    use_grn: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, H, W, C]
            train: Whether in training mode
        Returns:
            Output tensor of shape [B, H, W, C]
        """
        residual = x
        
        # 3x3 convolution
        x = nn.Conv(
            features=self.dim,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
        )(x)
        
        # LayerNorm
        x = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)(x)
        
        # Pointwise convolutions: expand -> GELU -> contract
        x = nn.Conv(features=2 * self.dim, kernel_size=(1, 1), use_bias=True)(x)
        x = jax.nn.gelu(x, approximate=True)
        
        # Optional Global Response Normalization
        if self.use_grn:
            x = GlobalResponseNormalization()(x)
        
        # Second pointwise convolution
        x = nn.Conv(features=self.dim, kernel_size=(1, 1), use_bias=True)(x)
        
        # Layer scale (optional)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                'layer_scale_gamma',
                lambda key, shape: jnp.ones(shape) * self.layer_scale_init_value,
                (self.dim,),
            )
            x = x * gamma  # Broadcasting: [C] -> [B, H, W, C]
        
        # Stochastic depth (drop path)
        if self.drop_path > 0.0 and train:
            keep_prob = 1.0 - self.drop_path
            random_tensor = jax.random.bernoulli(
                self.make_rng('drop_path'), keep_prob, x.shape[:1]
            )
            random_tensor = jnp.expand_dims(random_tensor, axis=tuple(range(1, x.ndim)))
            x = (x * random_tensor) / keep_prob
        
        return x + residual


# ============================================================================
# CONDITIONAL BLOCKS - Feature-wise modulation with conditioning
# ============================================================================


class ConditionalConvNeXtBlock(nn.Module):
    """
    Conditional ConvNeXt block with adaptive feature-wise modulation.
    
    Uses ConvNeXt blocks with FiLM-style conditioning for flow matching.
    """
    noise_dimension: int
    condition_dimension: int
    latent_dimension: int
    num_blocks: int
    image_size: int = 28
    use_grn: bool = True

    def setup(self):
        # Infer spatial dimensions from noise_dimension (assuming square images)
        self.spatial_size = int(math.sqrt(self.noise_dimension))
        self.channels = min(16, self.condition_dimension // 4)
        
        # Input projection with bottleneck
        bottleneck_dim = 128
        self.input_proj1 = nn.Dense(bottleneck_dim)
        self.input_proj2 = nn.Dense(
            self.spatial_size * self.spatial_size * self.channels
        )
        
        # ConvNeXt block for processing
        self.conv_block = ConvNeXtBlock(dim=self.channels, use_grn=self.use_grn)
        
        # Conditioning layer (FiLM-style modulation)
        self.conditioning_layer = nn.Dense(2 * self.channels)
        
        # Output projection with bottleneck
        bottleneck_dim = 128
        self.output_proj1 = nn.Dense(bottleneck_dim)
        self.output_proj2 = nn.Dense(self.noise_dimension)
        
        # LayerNorm for conditioning
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)

    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, noise_dimension]
            condition: Conditioning tensor of shape [B, condition_dimension]
        Returns:
            Output tensor of shape [B, noise_dimension]
        """
        residual = x
        
        # Project input to spatial representation
        # x: [B, noise_dim] -> [B, bottleneck] -> [B, H*W*C]
        x = self.input_proj1(x)
        x = jax.nn.gelu(x, approximate=True)
        x_spatial = self.input_proj2(x)
        # Reshape: [B, H*W*C] -> [B, H, W, C]
        x_spatial = x_spatial.reshape(
            x.shape[0], self.spatial_size, self.spatial_size, self.channels
        )
        
        # Apply LayerNorm
        x_spatial = self.ln(x_spatial)
        
        # Apply conditioning (FiLM: Feature-wise Linear Modulation)
        conditioning_params = self.conditioning_layer(condition)
        scale, shift = jnp.split(conditioning_params, 2, axis=-1)
        # Expand to spatial dimensions: [B, C] -> [B, 1, 1, C]
        scale = jnp.expand_dims(jnp.expand_dims(scale, 1), 1)
        shift = jnp.expand_dims(jnp.expand_dims(shift, 1), 1)
        x_spatial = (1.0 + scale) * x_spatial + shift
        
        # Apply ConvNeXt block
        x_spatial = self.conv_block(x_spatial)
        
        # Flatten back: [B, H, W, C] -> [B, H*W*C]
        x_flat = x_spatial.reshape(residual.shape[0], -1)
        
        # Project back to noise_dimension
        x = self.output_proj1(x_flat)
        x = jax.nn.gelu(x, approximate=True)
        x = self.output_proj2(x)
        
        # Residual connection with scaling
        return x / self.num_blocks + residual


# ============================================================================
# MAIN MODEL - Conditional ConvNeXt flow model
# ============================================================================


class ConditionalConvFlow(nn.Module):
    """
    Conditional Flow model using ConvNeXt-style architecture.
    
    Uses convolutional blocks with adaptive feature-wise modulation for flow matching.
    """
    noise_dimension: int
    condition_dimension: int
    num_blocks: int
    latent_dimension: int
    image_size: int = 28
    use_grn: bool = True
    num_latent_tokens: int = 32

    def setup(self):
        self.blocks = [
            ConditionalConvNeXtBlock(
                noise_dimension=self.noise_dimension,
                condition_dimension=self.condition_dimension,
                latent_dimension=self.latent_dimension,
                num_blocks=self.num_blocks,
                image_size=self.image_size,
                use_grn=self.use_grn,
            )
            for _ in range(self.num_blocks)
        ]
        # Project latents to condition dimension
        self.latent_proj = nn.Dense(self.condition_dimension)

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
        cond = t_emb + h_emb  # [B, condition_dim]
        
        # Add latent conditioning if provided
        if latents is not None:
            # Flatten latents: [B, num_latent_tokens, latent_dim] -> [B, num_latent_tokens * latent_dim]
            latents_flat = latents.reshape(latents.shape[0], -1)
            # Project to condition_dimension
            latent_cond = self.latent_proj(latents_flat)
            cond = cond + latent_cond
        
        # Apply conditional ConvNeXt blocks
        for blk in self.blocks:
            x = blk(x, cond)
        
        return x

