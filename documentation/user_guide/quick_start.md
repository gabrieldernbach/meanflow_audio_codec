# Quick Start

This guide will help you get started with the Meanflow Audio Codec project.

## Overview

This project builds an **Improved Mean Flow audio encoder** that operates in the MDCT domain. As a first step, we **benchmark the Improved Mean Flow method on MNIST** to validate the implementation before applying it to audio encoding tasks.

The implementation includes:

- **Improved Mean Flow (iMF) Audio Encoder**: Core architecture for audio encoding in MDCT domain
- **MNIST Benchmarking**: Initial validation of iMF on MNIST image generation
- **Conditional Flow Models**: Class-conditional generative models for both MNIST and audio
- **MDCT Preprocessing**: Modified Discrete Cosine Transform utilities for audio encoding
- **Baseline Flow Matching**: Standard Flow Matching implementation for comparison
- **Evaluation Tools**: Metrics, sampling, and classifier-based evaluation

## Benchmarking on MNIST

We first benchmark the Improved Mean Flow method on MNIST to validate the implementation. The main training functionality is available through the `train` module. Example usage:

```python
from pathlib import Path
from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.trainers.train import train_flow

config = TrainFlowConfig(
    batch_size=128,
    n_steps=10000,
    sample_every=1000,
    sample_seed=42,
    sample_steps=50,
    base_lr=1e-4,
    weight_decay=1e-4,
    seed=42,
    use_improved_mean_flow=True,  # Set to False for baseline Flow Matching
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

train_flow(config)
```

## Model Architecture

The `ConditionalFlow` model uses:
- **Conditional Residual Blocks**: Adaptive LayerNorm-style feature-wise modulation (AdaLN/FiLM variants)
- **Sinusoidal Time Embeddings**: For temporal conditioning
- **Class Embeddings**: For class-conditional generation

## Key Components

- **`ConditionalFlow`**: Main flow model with conditional residual blocks
- **`ConditionalResidualBlock`**: Adaptive normalization blocks with conditioning
- **`ConvNetClassifier`**: Classifier for evaluation metrics
- **Training Functions**: `train_step` (baseline) and `train_step_improved_mean_flow` (iMF)

## Testing

Run the test suite:

```bash
uv run pytest test/
```

Key test files:
- `test_improved_mean_flow.py`: Tests for Improved Mean Flow implementation
- `test_mdct.py`: MDCT forward/inverse transform and perfect reconstruction tests
- `test_mdct_perfect_reconstruction.py`: Round-trip MDCT reconstruction validation
- `test_mdct_reference.py`: Comparison with reference MDCT implementations
- `test_gelu.py`: GELU activation tests

## Next Steps

- See [Configuration](configuration.md) for detailed configuration options
- See [Examples](examples.md) for more usage examples
- See [Research Documentation](../research/README.md) for mathematical foundations

