# Configuration Reference

The Meanflow Audio Codec uses dataclass-based configuration for type safety and IDE support.

## TrainFlowConfig

The main configuration class for training flow models is `TrainFlowConfig`:

```python
from pathlib import Path
from meanflow_audio_codec.configs.config import TrainFlowConfig

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
```

## Configuration Parameters

### Training Parameters
- `batch_size`: Batch size for training
- `n_steps`: Total number of training steps
- `base_lr`: Base learning rate
- `weight_decay`: Weight decay for optimizer
- `seed`: Random seed for reproducibility

### Sampling Parameters
- `sample_every`: Frequency of sampling (in steps)
- `sample_seed`: Random seed for sampling
- `sample_steps`: Number of sampling steps

### Model Parameters
- `noise_dimension`: Dimension of noise/data space (e.g., 784 for MNIST 28x28)
- `condition_dimension`: Dimension of condition embedding
- `latent_dimension`: Dimension of latent space
- `num_blocks`: Number of residual blocks
- `num_classes`: Number of classes for conditional generation

### Training Options
- `use_improved_mean_flow`: Whether to use Improved Mean Flow (True) or baseline Flow Matching (False)
- `checkpoint_step`: Step interval for saving checkpoints (None = save at end)

### Paths
- `output_dir`: Base directory for outputs
- `run_name`: Name of the training run
- `data_dir`: Directory for dataset (None = use default)

## Usage

```python
from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.trainers.train import train_flow

config = TrainFlowConfig(
    batch_size=32,
    n_steps=5000,
    # ... other parameters
)

train_flow(config)
```

## See Also

- [Quick Start](quick_start.md) for basic usage examples
- [Examples](examples.md) for more configuration examples

