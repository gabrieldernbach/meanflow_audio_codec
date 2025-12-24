# API Reference

This document provides API reference for key components of the Meanflow Audio Codec.

## Training

### `train_flow(config: TrainFlowConfig)`

Main training function for flow models.

**Parameters:**
- `config`: `TrainFlowConfig` - Training configuration

**Example:**
```python
from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.trainers.train import train_flow

config = TrainFlowConfig(...)
train_flow(config)
```

## Models

### `ConditionalFlow`

Main flow model with conditional residual blocks.

**Location:** `meanflow_audio_codec.models.mlp_flow`

**Key Features:**
- Conditional residual blocks with adaptive normalization
- Sinusoidal time embeddings
- Class embeddings for conditional generation

### `SimpleConvFlow`

Simplified convolutional flow model.

**Location:** `meanflow_audio_codec.models.simple_conv_flow`

## MDCT

### `mdct(x, window_size=576, hop_size=None, use_fft_threshold=512)`

Forward MDCT transform. Automatically selects FFT-based (O(N log N)) or direct (O(N²)) implementation.

**Parameters:**
- `x`: `jnp.ndarray` - Input signal of shape `(..., T)`
- `window_size`: `int` - Size of MDCT window (default: 576)
- `hop_size`: `int | None` - Hop size between windows (default: None, uses `window_size // 2`)
- `use_fft_threshold`: `int` - Minimum window size for FFT implementation (default: 512)

**Returns:**
- `jnp.ndarray` - MDCT coefficients of shape `(..., n_frames, window_size)`

**Example:**
```python
from meanflow_audio_codec.preprocessing.mdct import mdct

audio_samples = ...  # Shape: (batch, time_samples)
mdct_coeffs = mdct(audio_samples, window_size=2048)
```

### `imdct(X, window_size=576, hop_size=None, use_fft_threshold=512)`

Inverse MDCT transform. Reconstructs audio from MDCT coefficients.

**Parameters:**
- `X`: `jnp.ndarray` - MDCT coefficients of shape `(..., n_frames, window_size)`
- `window_size`: `int` - Size of MDCT window (default: 576)
- `hop_size`: `int | None` - Hop size between windows (default: None, uses `window_size // 2`)
- `use_fft_threshold`: `int` - Minimum window size for FFT implementation (default: 512)

**Returns:**
- `jnp.ndarray` - Reconstructed audio of shape `(..., T)`

**Example:**
```python
from meanflow_audio_codec.preprocessing.mdct import imdct

mdct_coeffs = ...  # Shape: (batch, n_frames, window_size)
reconstructed_audio = imdct(mdct_coeffs, window_size=2048)
```

## Datasets

### `build_audio_pipeline(data_dir, ...)`

Build audio data loading pipeline.

**Location:** `meanflow_audio_codec.datasets.audio`

**Parameters:**
- `data_dir`: `str | Path` - Directory containing audio files
- Additional parameters for frame size, batch size, etc.

**Returns:**
- Iterator over audio batches

**Example:**
```python
from meanflow_audio_codec.datasets.audio import build_audio_pipeline

pipeline = build_audio_pipeline(data_dir="~/datasets/wavegen", ...)
for batch in pipeline:
    # Process batch
    pass
```

## Configuration

### `TrainFlowConfig`

Configuration dataclass for training flow models.

**Location:** `meanflow_audio_codec.configs.config`

**Key Parameters:**
- `batch_size`: `int` - Batch size
- `n_steps`: `int` - Number of training steps
- `base_lr`: `float` - Base learning rate
- `use_improved_mean_flow`: `bool` - Use Improved Mean Flow (True) or baseline (False)
- `noise_dimension`: `int` - Dimension of data space
- `condition_dimension`: `int` - Dimension of condition embedding
- `latent_dimension`: `int` - Dimension of latent space
- And more...

See [Configuration Reference](../user_guide/configuration.md) for complete parameter list.

## See Also

- [User Guide](../user_guide/README.md) for usage examples
- [Code Organization](code_organization.md) for code structure
- Source code for detailed docstrings

