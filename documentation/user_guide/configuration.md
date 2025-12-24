# Configuration Reference

The Meanflow Audio Codec uses a hierarchical dataclass-based configuration system with validation, schema generation, and migration support.

## Overview

The configuration system has been enhanced with:

- **Hierarchical structure**: Organized into logical sections (Base, Model, Dataset, Method, Training)
- **Strict validation**: Type checking, range validation, enum validation, and cross-field validation
- **Schema documentation**: Auto-generated schema and documentation from code
- **Config versioning**: Support for v1.0 (flat) and v2.0 (hierarchical) formats with automatic migration
- **Config merging**: Support for merging configs and factory functions for common configurations
- **Backward compatibility**: Properties provide flat access to nested config values

## TrainFlowConfig

The main configuration class for training flow models is `TrainFlowConfig`:

### Hierarchical Structure

```python
from pathlib import Path
from meanflow_audio_codec.configs.config import (
    TrainFlowConfig,
    BaseConfig,
    ModelConfig,
    DatasetConfig,
    MethodConfig,
    TrainingConfig,
)

config = TrainFlowConfig(
    base=BaseConfig(
        batch_size=128,
        n_steps=10000,
        base_lr=0.0001,
        weight_decay=0.0001,
        seed=42,
    ),
    model=ModelConfig(
        noise_dimension=784,
        condition_dimension=128,
        latent_dimension=256,
        num_blocks=8,
    ),
    dataset=DatasetConfig(
        dataset="mnist",
        tokenization_strategy="reshape",
    ),
    method=MethodConfig(
        use_improved_mean_flow=False,
        method="flow_matching",
    ),
    training=TrainingConfig(
        sample_every=1000,
        sample_seed=42,
        sample_steps=50,
        workdir=Path("./outputs/my_experiment"),
    ),
)
```

### Backward Compatibility

For backward compatibility, you can still access config values using flat property access:

```python
config.batch_size  # Accesses config.base.batch_size
config.workdir     # Accesses config.training.workdir
config.method      # Accesses config.method.method (method name)
config.use_improved_mean_flow  # Accesses config.method.use_improved_mean_flow
```

### Factory Functions

Use factory functions for common configurations:

```python
from meanflow_audio_codec.configs.config import create_mnist_config, create_audio_config

# Create MNIST config with defaults
mnist_config = create_mnist_config()

# Create audio config with defaults
audio_config = create_audio_config()

# Override specific values
custom_config = create_mnist_config(
    base={"batch_size": 256},
    training={"workdir": "./outputs/custom"},
)
```

## Configuration Sections

### BaseConfig

Core training parameters:

- `batch_size`: `int` - Batch size for training (must be > 0)
- `n_steps`: `int` - Total number of training steps (must be > 0)
- `base_lr`: `float` - Base learning rate (must be > 0)
- `weight_decay`: `float` - Weight decay for optimizer (must be >= 0)
- `seed`: `int` - Random seed for reproducibility

### ModelConfig

Model architecture parameters:

- `noise_dimension`: `int` - Dimension of noise/data space (must be > 0)
- `condition_dimension`: `int` - Dimension of condition embedding (must be > 0 and even)
- `latent_dimension`: `int` - Dimension of latent space (must be > 0)
- `num_blocks`: `int` - Number of residual blocks (must be > 0)
- `architecture`: `str | None` - Architecture type: "mlp", "mlp_mixer", or "convnet"

### DatasetConfig

Dataset-specific parameters:

- `dataset`: `str | None` - Dataset type: "mnist" or "audio"
- `data_dir`: `str | None` - Directory for dataset
- `tokenization_strategy`: `str | None` - Tokenization strategy: "mdct" or "reshape"
- `tokenization_config`: `dict | None` - Strategy-specific parameters

### MethodConfig

Method-specific parameters:

- `method`: `str | None` - Method type: "autoencoder", "flow_matching", "mean_flow", or "improved_mean_flow"
- `use_improved_mean_flow`: `bool` - Whether to use Improved Mean Flow (default: False)
- `gamma`: `float | None` - Mean flow gamma parameter (must be > 0)
- `flow_ratio`: `float | None` - Mean flow ratio parameter (must be > 0)
- `c`: `float | None` - Mean flow c parameter (must be > 0)
- `use_stop_gradient`: `bool | None` - Whether to use stop gradient
- `loss_weighting`: `str | None` - Loss weighting: "uniform", "time_dependent", or "learned"
- `loss_strategy`: `str | None` - Loss strategy: "flow_matching", "mean_flow", or "improved_mean_flow"
- `noise_schedule`: `str | None` - Noise schedule: "linear" or "uniform"
- `time_sampling`: `str | None` - Time sampling: "uniform", "logit_normal", or "mean_flow"
- Additional time sampling and noise schedule parameters

### TrainingConfig

Training infrastructure parameters:

- `sample_every`: `int` - Frequency of sampling (in steps, must be > 0)
- `sample_seed`: `int` - Random seed for sampling
- `sample_steps`: `int` - Number of sampling steps (must be > 0)
- `workdir`: `Path | None` - Working directory for outputs (required)
- `checkpoint_step`: `int | None` - Step interval for saving checkpoints (must be > 0 if provided)

## Loading Configurations

### From JSON File

```python
from pathlib import Path
from meanflow_audio_codec.configs.config import load_config_from_json

# Automatically detects and migrates v1.0 (flat) configs to v2.0 (hierarchical)
config = load_config_from_json(Path("configs/my_config.json"))
```

### Config Migration

The system automatically migrates v1.0 (flat) configs to v2.0 (hierarchical) format:

```python
from meanflow_audio_codec.configs.config import migrate_config_v1_to_v2

v1_config = {
    "batch_size": 128,
    "n_steps": 10000,
    # ... other flat fields
}

v2_config = migrate_config_v1_to_v2(v1_config)
# Returns hierarchical structure with config_version="2.0"
```

You can also use the migration script:

```bash
uv run python meanflow_audio_codec/tools/migrate_configs.py configs/old_config.json -o configs/new_config.json
```

## Config Validation

All configs are validated automatically when created:

```python
from meanflow_audio_codec.configs.config import BaseConfig

# This will raise ValueError
try:
    config = BaseConfig(
        batch_size=0,  # Invalid: must be > 0
        n_steps=10000,
        base_lr=0.0001,
        weight_decay=0.0001,
        seed=42,
    )
    config.validate()
except ValueError as e:
    print(f"Validation error: {e}")
```

## Config Merging

Merge configurations for experiments:

```python
from meanflow_audio_codec.configs.config import create_mnist_config, merge_configs

base_config = create_mnist_config()
override = {
    "base": {"batch_size": 256},
    "training": {"sample_every": 500},
}

merged_config = merge_configs(base_config, override)
```

## Config Diff

Compare configurations:

```python
from meanflow_audio_codec.configs.config import create_mnist_config, diff_configs, print_config_diff

config1 = create_mnist_config()
config2 = create_mnist_config()
config2.base.batch_size = 256

diff = diff_configs(config1, config2)
print_config_diff(diff)
# Output:
# Changed parameters:
#   base.batch_size: 128 -> 256
```

## Schema and Documentation

Generate schema and documentation:

```python
config = create_mnist_config()

# Get schema (dict with field metadata)
schema = config.get_schema()

# Get human-readable documentation
docs = config.get_documentation()
print(docs)
```

## Usage in Training

```python
from meanflow_audio_codec.configs.config import load_config_from_json
from meanflow_audio_codec.trainers.train import train_flow
from pathlib import Path

config = load_config_from_json(Path("configs/my_config.json"))
train_flow(config, resume=False)
```

## Migration from v1.0

Existing v1.0 (flat) configs are automatically migrated when loaded. To migrate all configs in a directory:

```bash
uv run python meanflow_audio_codec/tools/migrate_configs.py configs/ -o configs_migrated/
```

The migration preserves all values and converts the flat structure to the hierarchical v2.0 format.
