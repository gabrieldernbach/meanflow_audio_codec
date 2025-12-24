# Usage Examples

This document provides examples of how to use the Meanflow Audio Codec project.

## Basic Training Example

```python
from pathlib import Path
from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.trainers.train import train_flow

# Create configuration
config = TrainFlowConfig(
    batch_size=128,
    n_steps=10000,
    sample_every=1000,
    sample_seed=42,
    sample_steps=50,
    base_lr=1e-4,
    weight_decay=1e-4,
    seed=42,
    use_improved_mean_flow=True,
    output_dir=Path("./outputs"),
    run_name="my_experiment",
    checkpoint_step=None,
    data_dir=None,
    noise_dimension=784,  # 28x28 for MNIST
    condition_dimension=128,
    latent_dimension=256,
    num_blocks=8,
    num_classes=10,
)

# Start training
train_flow(config)
```

## MDCT Usage Example

```python
from meanflow_audio_codec.preprocessing.mdct import mdct, imdct

# Transform audio to MDCT domain
audio_samples = ...  # Shape: (batch, time_samples)
mdct_coeffs = mdct(audio_samples, window_size=2048)  # Shape: (batch, num_frames, window_size)

# Reconstruct audio from MDCT coefficients
reconstructed_audio = imdct(mdct_coeffs, window_size=2048)
```

## Audio Encoder Pipeline

The audio encoder operates as follows:

1. **Audio → MDCT**: Transform input audio to MDCT spectral domain
2. **MDCT → Latent**: Encode MDCT coefficients using Improved Mean Flow
3. **Latent → MDCT**: Decode latent codes back to MDCT coefficients
4. **MDCT → Audio**: Reconstruct audio using inverse MDCT

This approach follows the MDCTCodec architecture (see [MDCTCodec Key Equations](../research/mdct/mdctcodec_key_eqn.md)), where the entire encoding/decoding process operates in the MDCT domain rather than directly on waveforms.

## Baseline Flow Matching

To use baseline Flow Matching instead of Improved Mean Flow:

```python
config = TrainFlowConfig(
    # ... other parameters
    use_improved_mean_flow=False,  # Use baseline Flow Matching
)
```

## See Also

- [Quick Start](quick_start.md) for getting started
- [Configuration](configuration.md) for detailed configuration options
- [Research Documentation](../research/README.md) for mathematical foundations

