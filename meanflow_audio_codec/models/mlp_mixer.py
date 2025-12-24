import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from meanflow_audio_codec.utils import sinusoidal_embedding

# ============================================================================
# BUILDING BLOCKS - MLP-Mixer components
# ============================================================================


class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer block with token mixing and channel mixing.
    
    Architecture: TokenMix -> ChannelMix with AdaLN and residual connections.
    Uses adaptive layer normalization conditioned on diffusion time.
    """
    token_mix_dim: int
    channel_mix_dim: int
    num_channels: int
    num_tokens: int
    condition_dim: int

    def _apply_adaln(
        self, x: jnp.ndarray, condition: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply adaptive layer normalization (AdaLN).

        Args:
            x: Input tensor of shape [B, num_tokens, num_channels]
            condition: Conditioning tensor of shape [B, condition_dim]
        Returns:
            Normalized and modulated tensor of same shape
        """
        x = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)
        # Generate scale and shift from condition
        scale_shift = nn.Dense(2 * self.num_channels)(condition)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        # Expand to match spatial dimensions: [B, C] -> [B, 1, C]
        scale = jnp.expand_dims(scale, 1)
        shift = jnp.expand_dims(shift, 1)
        return (1.0 + scale) * x + shift

    def _apply_mlp(
        self, x: jnp.ndarray, hidden_dim: int, out_dim: int
    ) -> jnp.ndarray:
        """
        Apply MLP: Dense -> GELU -> Dense.

        Args:
            x: Input tensor
            hidden_dim: Hidden dimension for the first Dense layer
            out_dim: Output dimension for the second Dense layer
        Returns:
            Output tensor after MLP transformation
        """
        x = nn.Dense(hidden_dim)(x)
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Dense(out_dim)(x)
        return x

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, condition: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, num_tokens, num_channels]
            condition: Conditioning tensor of shape [B, condition_dim]
        Returns:
            Output tensor of shape [B, num_tokens, num_channels]
        """
        residual = x

        # Token mixing: mix across spatial/token dimension
        x = self._apply_adaln(x, condition)
        # [B, num_tokens, num_channels] -> [B, num_channels, num_tokens]
        x = x.transpose(0, 2, 1)
        x = self._apply_mlp(x, self.token_mix_dim, self.num_tokens)
        # [B, num_channels, num_tokens] -> [B, num_tokens, num_channels]
        x = x.transpose(0, 2, 1)
        x = x + residual

        # Channel mixing: mix across channel dimension
        residual = x
        x = self._apply_adaln(x, condition)
        x = self._apply_mlp(x, self.channel_mix_dim, self.num_channels)
        x = x + residual

        return x


# ============================================================================
# CONDITIONAL BLOCKS - Feature-wise modulation with conditioning
# ============================================================================


class ConditionalMLPMixerBlock(nn.Module):
    """
    Conditional MLP-Mixer block with adaptive feature-wise modulation.
    
    Similar to ConditionalResidualBlock but uses MLP-Mixer architecture.
    """
    noise_dimension: int
    condition_dimension: int
    latent_dimension: int
    num_blocks: int
    token_mix_dim: int = 2048
    channel_mix_dim: int = 2048
    num_channels: int = 16

    def setup(self):
        # Infer spatial dimensions from noise_dimension
        # (assuming square images)
        self.spatial_size = int(math.sqrt(self.noise_dimension))
        self.num_tokens = self.spatial_size * self.spatial_size

        # Projection to spatial representation with channels
        self.input_proj = nn.Dense(self.num_tokens * self.num_channels)

        # MLP-Mixer block with conditional AdaLN support
        self.mixer_block = MLPMixerBlock(
            token_mix_dim=self.token_mix_dim,
            channel_mix_dim=self.channel_mix_dim,
            num_channels=self.num_channels,
            num_tokens=self.num_tokens,
            condition_dim=self.condition_dimension,
        )

        # Output projection
        self.output_proj = nn.Dense(self.noise_dimension)

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
        # x: [B, noise_dim] -> [B, num_tokens * num_channels]
        x = self.input_proj(x)
        # Reshape: [B, num_tokens * num_channels] -> [B, num_tokens, C]
        x = x.reshape(x.shape[0], self.num_tokens, self.num_channels)

        # Apply MLP-Mixer block with conditioning (AdaLN applied internally)
        x = self.mixer_block(x, condition=condition)

        # Flatten: [B, num_tokens, num_channels] -> [B, num_tokens * C]
        x_flat = x.reshape(residual.shape[0], -1)

        # Project back to noise_dimension
        x = self.output_proj(x_flat)

        # Residual connection with scaling
        return x / self.num_blocks + residual


# ============================================================================
# MAIN MODEL - Conditional MLP-Mixer flow model
# ============================================================================


class ConditionalMLPMixerFlow(nn.Module):
    """
    Conditional Flow model using MLP-Mixer architecture.
    
    Stacks ConditionalMLPMixerBlock layers for flow matching.
    """
    noise_dimension: int
    condition_dimension: int
    num_blocks: int
    latent_dimension: int
    token_mix_dim: int = 2048
    channel_mix_dim: int = 2048
    num_channels: int = 16
    num_latent_tokens: int = 32

    def setup(self):
        self.blocks = [
            ConditionalMLPMixerBlock(
                noise_dimension=self.noise_dimension,
                condition_dimension=self.condition_dimension,
                latent_dimension=self.latent_dimension,
                num_blocks=self.num_blocks,
                token_mix_dim=self.token_mix_dim,
                channel_mix_dim=self.channel_mix_dim,
                num_channels=self.num_channels,
            )
            for _ in range(self.num_blocks)
        ]
        # Project latents to condition dimension
        self.latent_proj = nn.Dense(self.condition_dimension)

    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        latents: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, noise_dimension]
            time: Time tensor of shape [B, 2] (t and h)
            latents: Optional latent tokens of shape
                [B, num_latent_tokens, latent_dim].
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
            # Flatten: [B, num_latent_tokens, latent_dim] -> [B, ...]
            latents_flat = latents.reshape(latents.shape[0], -1)
            # Project to condition_dimension
            latent_cond = self.latent_proj(latents_flat)
            cond = cond + latent_cond

        # Apply conditional MLP-Mixer blocks
        for blk in self.blocks:
            x = blk(x, cond)

        return x


# ============================================================================
# AUTOENCODER COMPONENTS - Encoder and decoder for latent extraction
# ============================================================================


class MLPMixerEncoder(nn.Module):
    """
    MLP-Mixer Encoder that extracts information from input into latent tokens.
    
    Uses learnable query tokens to aggregate information from context.
    """
    input_dim: int
    num_latent_tokens: int = 32
    latent_dim: int = 512
    num_context_tokens: int = 512
    token_mix_dim: int = 2048
    channel_mix_dim: int = 2048

    def setup(self):
        # Project input to context tokens
        self.input_proj = nn.Dense(self.num_context_tokens * self.latent_dim)

        # Learnable query tokens for latent representation
        self.latent_queries = self.param(
            'latent_queries',
            nn.initializers.normal(stddev=0.02),
            (self.num_latent_tokens, self.latent_dim),
        )

        # Learnable condition embedding for AdaLN
        condition_dim = self.latent_dim  # Use latent_dim as condition dimension
        self.condition_emb = self.param(
            'condition_emb',
            nn.initializers.normal(stddev=0.02),
            (condition_dim,),
        )

        # MLP-Mixer block for extraction
        total_tokens = self.num_context_tokens + self.num_latent_tokens
        self.mixer_block = MLPMixerBlock(
            token_mix_dim=self.token_mix_dim,
            channel_mix_dim=self.channel_mix_dim,
            num_channels=self.latent_dim,
            num_tokens=total_tokens,
            condition_dim=condition_dim,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, input_dim]
        Returns:
            Latent tokens of shape [B, num_latent_tokens, latent_dim]
        """
        batch_size = x.shape[0]

        # Project input to context tokens
        # x: [B, input_dim] -> [B, num_context_tokens * latent_dim]
        x_proj = self.input_proj(x)
        # Reshape: [B, num_context_tokens * latent_dim] -> [B, ..., latent_dim]
        context_tokens = x_proj.reshape(
            batch_size, self.num_context_tokens, self.latent_dim
        )

        # Expand learnable query tokens to batch dimension
        # [num_latent_tokens, latent_dim] -> [B, num_latent_tokens, latent_dim]
        latent_queries = jnp.expand_dims(self.latent_queries, 0)
        latent_queries = jnp.broadcast_to(
            latent_queries,
            (batch_size, self.num_latent_tokens, self.latent_dim),
        )

        # Concatenate context tokens and latent query tokens
        # [B, num_context_tokens + num_latent_tokens, latent_dim]
        all_tokens = jnp.concatenate([context_tokens, latent_queries], axis=1)

        # Create condition from learnable embedding: [condition_dim] -> [B, condition_dim]
        condition = jnp.broadcast_to(self.condition_emb[None, :], (batch_size, self.condition_emb.shape[0]))

        # Apply MLP-Mixer block to aggregate information
        all_tokens = self.mixer_block(all_tokens, condition)

        # Extract the latent tokens (last num_latent_tokens)
        latent_tokens = all_tokens[:, self.num_context_tokens:, :]

        return latent_tokens


class MLPMixerDecoder(nn.Module):
    """
    MLP-Mixer Decoder that reconstructs output from latent tokens.
    
    Uses learnable query tokens to generate output representation.
    """
    output_dim: int
    num_latent_tokens: int = 32
    latent_dim: int = 512
    num_output_tokens: int = 512
    token_mix_dim: int = 2048
    channel_mix_dim: int = 2048

    def setup(self):
        # Learnable output query tokens
        self.output_queries = self.param(
            'output_queries',
            nn.initializers.normal(stddev=0.02),
            (self.num_output_tokens, self.latent_dim),
        )

        # Learnable condition embedding for AdaLN
        condition_dim = self.latent_dim  # Use latent_dim as condition dimension
        self.condition_emb = self.param(
            'condition_emb',
            nn.initializers.normal(stddev=0.02),
            (condition_dim,),
        )

        # MLP-Mixer block for decoding
        total_tokens = self.num_latent_tokens + self.num_output_tokens
        self.mixer_block = MLPMixerBlock(
            token_mix_dim=self.token_mix_dim,
            channel_mix_dim=self.channel_mix_dim,
            num_channels=self.latent_dim,
            num_tokens=total_tokens,
            condition_dim=condition_dim,
        )

        # Project output tokens to final output dimension
        self.output_proj = nn.Dense(self.output_dim)

    def __call__(self, latent_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            latent_tokens: Latent tokens of shape
                [B, num_latent_tokens, latent_dim]
        Returns:
            Output tensor of shape [B, output_dim]
        """
        batch_size = latent_tokens.shape[0]

        # Expand learnable output query tokens to batch dimension
        # [num_output_tokens, latent_dim] -> [B, num_output_tokens, latent_dim]
        output_queries = jnp.expand_dims(self.output_queries, 0)
        output_queries = jnp.broadcast_to(
            output_queries,
            (batch_size, self.num_output_tokens, self.latent_dim),
        )

        # Concatenate latent tokens and output query tokens
        # [B, num_latent_tokens + num_output_tokens, latent_dim]
        all_tokens = jnp.concatenate([latent_tokens, output_queries], axis=1)

        # Create condition from learnable embedding: [condition_dim] -> [B, condition_dim]
        condition = jnp.broadcast_to(self.condition_emb[None, :], (batch_size, self.condition_emb.shape[0]))

        # Apply MLP-Mixer block to decode
        all_tokens = self.mixer_block(all_tokens, condition)

        # Extract the output tokens (last num_output_tokens)
        output_tokens = all_tokens[:, self.num_latent_tokens:, :]

        # Flatten: [B, num_output_tokens, latent_dim] -> [B, ...]
        output_flat = output_tokens.reshape(batch_size, -1)

        # Project to final output dimension
        output = self.output_proj(output_flat)

        return output


class MLPMixerAutoencoder(nn.Module):
    """
    MLP-Mixer Autoencoder combining encoder and decoder.
    
    Encodes input into latent tokens, then decodes to output.
    """
    input_dim: int
    num_latent_tokens: int = 32
    latent_dim: int = 512
    num_context_tokens: int = 512
    num_output_tokens: int = 512
    token_mix_dim: int = 2048
    channel_mix_dim: int = 2048

    def setup(self):
        self.encoder = MLPMixerEncoder(
            input_dim=self.input_dim,
            num_latent_tokens=self.num_latent_tokens,
            latent_dim=self.latent_dim,
            num_context_tokens=self.num_context_tokens,
            token_mix_dim=self.token_mix_dim,
            channel_mix_dim=self.channel_mix_dim,
        )
        self.decoder = MLPMixerDecoder(
            output_dim=self.input_dim,
            num_latent_tokens=self.num_latent_tokens,
            latent_dim=self.latent_dim,
            num_output_tokens=self.num_output_tokens,
            token_mix_dim=self.token_mix_dim,
            channel_mix_dim=self.channel_mix_dim,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape [B, input_dim]
        Returns:
            Reconstructed output tensor of shape [B, input_dim]
        """
        # Encode to latent tokens
        latent_tokens = self.encoder(x)

        # Decode from latent tokens
        output = self.decoder(latent_tokens)

        return output

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Encode input to latent tokens.
        
        Args:
            x: Input tensor of shape [B, input_dim]
        Returns:
            Latent tokens of shape [B, num_latent_tokens, latent_dim]
        """
        return self.encoder(x)

    def decode(self, latent_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent tokens to output.
        
        Args:
            latent_tokens: Latent tokens of shape
                [B, num_latent_tokens, latent_dim]
        Returns:
            Output tensor of shape [B, input_dim]
        """
        return self.decoder(latent_tokens)
