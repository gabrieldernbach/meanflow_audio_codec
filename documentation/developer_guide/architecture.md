# Architecture Overview

This document provides a high-level overview of the Meanflow Audio Codec architecture.

## Project Structure

```
meanflow_audio_codec/
├── meanflow_audio_codec/          # Main package
│   ├── models/             # Flow models (for audio encoding) and classifiers
│   ├── trainers/           # Training loops and utilities
│   ├── evaluators/         # Sampling and metrics
│   ├── datasets/           # Data loading (MNIST for benchmarking)
│   ├── preprocessing/      # MDCT utilities for audio encoding
│   └── configs/           # Configuration dataclasses
├── test/                   # Unit tests (including MDCT tests)
├── documentation/          # Research notes and equations
└── pyproject.toml          # Project configuration
```

## Core Components

### Models (`models/`)

- **Flow Models**: Generative models for encoding/decoding
  - `mlp_flow.py`: MLP-based flow models
  - `conv_flow.py`: Convolutional flow models
  - `simple_conv_flow.py`: Simplified convolutional flow models
- **TrainState**: Custom training state management
- **Classifiers**: Models for evaluation metrics

### Trainers (`trainers/`)

- **Training Loops**: Main training orchestration
  - `train.py`: Main training loop
  - `training_steps.py`: Individual training step functions
- **Utilities**: Helper functions for training
  - `utils.py`: Logging, checkpointing, visualization

### Evaluators (`evaluators/`)

- **Sampling**: Generation and sampling functions
  - `sampling.py`: Sampling from trained models
- **Metrics**: Evaluation metrics
  - `metrics.py`: Metrics computation

### Datasets (`datasets/`)

- **Data Loaders**: Generic dataset loaders
  - `mnist.py`: MNIST dataset loading
  - `audio.py`: Audio dataset loading

### Preprocessing (`preprocessing/`)

- **MDCT**: Modified Discrete Cosine Transform
  - `mdct.py`: Forward and inverse MDCT transforms

### Configs (`configs/`)

- **Configuration**: Type-safe configuration dataclasses
  - `config.py`: Training and model configuration classes

## Audio Encoder Pipeline

The audio encoder operates entirely in the MDCT domain:

1. **Audio → MDCT**: Transform input audio to MDCT spectral domain
2. **MDCT → Latent**: Encode MDCT coefficients using Improved Mean Flow
3. **Latent → MDCT**: Decode latent codes back to MDCT coefficients
4. **MDCT → Audio**: Reconstruct audio using inverse MDCT

This approach follows the MDCTCodec architecture, where the entire encoding/decoding process operates in the MDCT domain rather than directly on waveforms.

## Training Flow

1. **Data Loading**: Load and preprocess data (MNIST or audio)
2. **Model Initialization**: Initialize flow model with configuration
3. **Training Loop**: Iterate over training steps
   - Forward pass through model
   - Compute loss (Improved Mean Flow or baseline Flow Matching)
   - Backward pass and optimization
   - Logging and checkpointing
4. **Evaluation**: Sample from model and compute metrics

## Key Design Decisions

1. **MDCT Domain Processing**: Operates in spectral domain for efficiency
2. **Improved Mean Flow**: Uses iMF for fast one-step generation
3. **Type-Safe Configuration**: Dataclass-based configs for IDE support
4. **Modular Design**: Clear separation of concerns across components

## See Also

- [Code Organization](code_organization.md) for detailed code structure
- [Research Documentation](../research/README.md) for mathematical foundations
- [User Guide](../user_guide/README.md) for usage instructions

