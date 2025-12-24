"""Factory functions for creating flow models.

Provides convenient factory functions for common model compositions,
enabling easy model creation from configurations.
"""

from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.models.conv_flow import ConditionalConvFlow
from meanflow_audio_codec.models.mlp_flow import ConditionalFlow
from meanflow_audio_codec.models.mlp_mixer import ConditionalMLPMixerFlow


def create_mlp_flow(
    noise_dimension: int,
    latent_dimension: int,
    num_blocks: int,
    condition_dimension: int,
) -> ConditionalFlow:
    """Create a ConditionalFlow model with MLP architecture.
    
    Args:
        noise_dimension: Dimension of noise/data space
        latent_dimension: Dimension of latent space
        num_blocks: Number of conditional residual blocks
        condition_dimension: Dimension of condition embedding
    
    Returns:
        ConditionalFlow model
    """
    return ConditionalFlow(
        noise_dimension=noise_dimension,
        condition_dimension=condition_dimension,
        num_blocks=num_blocks,
        latent_dimension=latent_dimension,
    )


def create_conv_flow(
    noise_dimension: int,
    latent_dimension: int,
    num_blocks: int,
    condition_dimension: int,
    image_size: int = 28,
    base_channels: int = 64,
) -> ConditionalConvFlow:
    """Create a ConditionalConvFlow model with ConvNeXt architecture.
    
    Args:
        noise_dimension: Dimension of noise/data space
        latent_dimension: Dimension of latent space
        num_blocks: Number of conditional ConvNeXt blocks
        condition_dimension: Dimension of condition embedding
        image_size: Spatial size of input (assumes square)
        base_channels: Base number of channels
    
    Returns:
        ConditionalConvFlow model
    """
    return ConditionalConvFlow(
        noise_dimension=noise_dimension,
        condition_dimension=condition_dimension,
        num_blocks=num_blocks,
        latent_dimension=latent_dimension,
        image_size=image_size,
        base_channels=base_channels,
    )


def create_mlp_mixer_flow(
    noise_dimension: int,
    latent_dimension: int,
    num_blocks: int,
    condition_dimension: int,
    token_mix_dim: int = 2048,
    channel_mix_dim: int = 2048,
    num_channels: int = 16,
    num_latent_tokens: int = 32,
) -> ConditionalMLPMixerFlow:
    """Create a ConditionalMLPMixerFlow model.
    
    Args:
        noise_dimension: Dimension of noise/data space
        latent_dimension: Dimension of latent space
        num_blocks: Number of conditional MLP-Mixer blocks
        condition_dimension: Dimension of condition embedding
        token_mix_dim: Dimension for token mixing MLP
        channel_mix_dim: Dimension for channel mixing MLP
        num_channels: Number of channels in token representation
        num_latent_tokens: Number of latent tokens
    
    Returns:
        ConditionalMLPMixerFlow model
    """
    return ConditionalMLPMixerFlow(
        noise_dimension=noise_dimension,
        condition_dimension=condition_dimension,
        num_blocks=num_blocks,
        latent_dimension=latent_dimension,
        token_mix_dim=token_mix_dim,
        channel_mix_dim=channel_mix_dim,
        num_channels=num_channels,
        num_latent_tokens=num_latent_tokens,
    )


def create_flow_model(config: TrainFlowConfig):
    """Create a flow model from configuration.
    
    Args:
        config: Training configuration
    
    Returns:
        Flow model (ConditionalFlow, ConditionalConvFlow, or ConditionalMLPMixerFlow)
    
    Raises:
        ValueError: If architecture is not recognized
    """
    architecture = config.architecture or "mlp"
    
    if architecture == "mlp":
        return create_mlp_flow(
            noise_dimension=config.noise_dimension,
            latent_dimension=config.latent_dimension,
            num_blocks=config.num_blocks,
            condition_dimension=config.condition_dimension,
        )
    elif architecture == "convnet":
        # Infer image size from noise_dimension (assumes square)
        image_size = int(config.noise_dimension ** 0.5)
        return create_conv_flow(
            noise_dimension=config.noise_dimension,
            latent_dimension=config.latent_dimension,
            num_blocks=config.num_blocks,
            condition_dimension=config.condition_dimension,
            image_size=image_size,
        )
    elif architecture == "mlp_mixer":
        return create_mlp_mixer_flow(
            noise_dimension=config.noise_dimension,
            latent_dimension=config.latent_dimension,
            num_blocks=config.num_blocks,
            condition_dimension=config.condition_dimension,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            "Must be one of: 'mlp', 'convnet', 'mlp_mixer'"
        )

