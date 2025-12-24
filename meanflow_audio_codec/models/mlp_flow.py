import jax
import jax.numpy as jnp
from flax import linen as nn

from meanflow_audio_codec.utils import sinusoidal_embedding

# ============================================================================
# BUILDING BLOCKS - Reusable components
# ============================================================================


class MLP(nn.Module):
    """Simple two-layer MLP with GELU activation."""
    hidden: int
    out: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden)
        self.dense2 = nn.Dense(features=self.out)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, input_dim]
        Returns:
            Output tensor of shape [B, out]
        """
        x = self.dense1(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.dense2(x)
        return x


# ============================================================================
# ENCODER - Image compression
# ============================================================================


class MLPEncoder(nn.Module):
    """Encoder that compresses full-resolution image to lower-dimensional latent."""  # noqa: E501
    input_dimension: int
    latent_dimension: int

    def setup(self):
        hidden_dim = (self.input_dimension + self.latent_dimension) // 2
        self.encoder_mlp = MLP(hidden_dim, self.latent_dimension)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, input_dimension]
        Returns:
            Latent tensor of shape [B, latent_dimension]
        """
        return self.encoder_mlp(x)


# ============================================================================
# CONDITIONAL BLOCKS - Feature-wise modulation with conditioning
# ============================================================================


class ConditionalResidualBlock(nn.Module):
    """
    Adaptive LayerNorm-style feature-wise modulation (AdaLN / FiLM).
    
    Takes latent+noise concatenated as input and applies adaptive normalization
    conditioned on time embeddings. Reference: https://arxiv.org/pdf/2212.09748
    """
    input_dimension: int
    noise_dimension: int
    condition_dimension: int
    num_blocks: int

    def setup(self):
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)
        self.conditioning_layer = MLP(
            self.condition_dimension,
            2 * self.input_dimension + self.noise_dimension,
        )
        self.mlp = MLP(self.input_dimension, self.noise_dimension)

    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, input_dimension]
                (latent + noise concatenated)
            condition: Conditioning tensor of shape [B, condition_dimension]
        Returns:
            Output tensor of shape [B, noise_dimension]
        """
        # Residual connection from noise part only
        residual = x[:, -self.noise_dimension:]
        
        # Adaptive normalization
        x = self.ln(x)
        scale_shift_scale = self.conditioning_layer(condition)
        
        # Split into scale1, shift, and scale2
        # scale_shift_scale: [B, 2*input_dim + noise_dim]
        scale1_shift, scale2 = jnp.split(
            scale_shift_scale,
            [2 * self.input_dimension],
            axis=-1,
        )
        scale1, shift = jnp.split(scale1_shift, 2, axis=-1)
        
        # Apply adaptive modulation and MLP
        mlp_out = self.mlp((1.0 + scale1) * x + shift)
        
        # Enforce output dimension
        if mlp_out.shape[-1] != self.noise_dimension:
            mlp_out = mlp_out[..., :self.noise_dimension]
        
        # Apply output scale and residual
        x = mlp_out * (1.0 + scale2)
        return x / self.num_blocks + residual


# ============================================================================
# MAIN MODEL - Autoencoder flow model
# ============================================================================


class ConditionalFlow(nn.Module):
    """
    Autoencoder flow model.
    
    Encoder compresses image to latent, decoder denoises from latent + noise
    concatenated to full resolution.
    """
    noise_dimension: int
    condition_dimension: int
    num_blocks: int
    latent_dimension: int

    def setup(self):
        self.encoder = MLPEncoder(
            input_dimension=self.noise_dimension,
            latent_dimension=self.latent_dimension,
        )
        input_dim = self.latent_dimension + self.noise_dimension
        self.blocks = [
            ConditionalResidualBlock(
                input_dimension=input_dim,
                noise_dimension=self.noise_dimension,
                condition_dimension=self.condition_dimension,
                num_blocks=self.num_blocks,
            )
            for _ in range(self.num_blocks)
        ]

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encode full-resolution image to latent.
        
        Args:
            x: Input tensor of shape [B, noise_dimension]
        Returns:
            Latent tensor of shape [B, latent_dimension]
        """
        return self.encoder(x)

    def _decode(self, x_concat: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        """
        Internal decode method with concatenated input.
        
        Args:
            x_concat: Concatenated tensor of shape
                [B, latent_dimension + noise_dimension]
            time: Time tensor of shape [B, 2] (t and h)
        Returns:
            Output tensor of shape [B, noise_dimension]
        """
        # Extract latent part from concatenated input
        # This will be re-concatenated with each block's output
        latent = x_concat[:, :self.latent_dimension]
        noise = x_concat[:, self.latent_dimension:]
        
        # Time embeddings: [B, 2] -> [B, condition_dim] each
        t_emb = sinusoidal_embedding(time[:, 0], self.condition_dimension)
        h_emb = sinusoidal_embedding(time[:, 1], self.condition_dimension)
        cond = t_emb + h_emb  # [B, condition_dim]
        
        # Apply decoder blocks
        # Blocks expect input_dimension = latent_dimension + noise_dimension
        # Blocks output [B, noise_dimension]
        # After each block, re-concatenate latent with output for next block
        x = noise
        for blk in self.blocks:
            # Concatenate latent with current noise output
            x_concat = jnp.concatenate([latent, x], axis=-1)
            # Process through block
            x = blk(x_concat, cond)
        
        # Final output is [B, noise_dimension]
        return x

    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        latents: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Decode: denoise from latent + noise concatenated to full resolution.
        
        Args:
            x: Noise tensor of shape [B, noise_dimension]
            time: Time tensor of shape [B, 2] (t and h)
            latents: Latent tensor of shape [B, latent_dimension].
                     If None, unconditional mode
                     (for classifier-free guidance).
        Returns:
            Output tensor of shape [B, noise_dimension]
        """
        # Concatenate latent with noise
        if latents is not None:
            # x: [B, noise_dim], latents: [B, latent_dim]
            # x: [B, noise_dim], latents: [B, latent_dim]
            # Concatenate: [B, latent_dim + noise_dim]
            x_concat = jnp.concatenate([latents, x], axis=-1)
        else:
            # Unconditional: use zero latents
            zero_latents = jnp.zeros(
                (x.shape[0], self.latent_dimension), dtype=x.dtype
            )
            x_concat = jnp.concatenate([zero_latents, x], axis=-1)
        
        return self._decode(x_concat, time)
