# Composability Guide

This guide explains how to use the composable components in the codebase to create custom training configurations, loss functions, noise schedules, and preprocessing pipelines.

## Overview

The codebase uses a strategy pattern for composability, allowing you to:
- Swap loss functions independently
- Mix and match noise schedules and time sampling strategies
- Compose preprocessing pipelines
- Create models using factory functions

## Loss Strategies

Loss strategies encapsulate the complete loss computation logic, including noise sampling, time sampling, target computation, and loss calculation.

### Available Loss Strategies

- **FlowMatchingLoss**: Standard flow matching loss
- **MeanFlowLoss**: Mean flow with adaptive reweighting
- **ImprovedMeanFlowLoss**: Improved mean flow with JVP and network-independent target

### Creating a Custom Loss Strategy

```python
from meanflow_audio_codec.trainers.loss_strategies import LossStrategy
import jax
import jax.numpy as jnp
from meanflow_audio_codec.models import TrainState

class CustomLossStrategy(LossStrategy):
    def __init__(self, noise_schedule, time_sampling, use_weighted_loss=True):
        self.noise_schedule = noise_schedule
        self.time_sampling = time_sampling
        self.use_weighted_loss = use_weighted_loss
    
    def compute_loss(self, state, key, x):
        # Your custom loss computation logic
        key, k_noise, k_time = jax.random.split(key, 3)
        noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
        time = self.time_sampling.sample_time(k_time, x.shape[0], dtype=x.dtype)
        
        # ... rest of your implementation
        
        def loss_fn(params):
            # Compute loss
            return loss_value
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, grads
```

### Using Loss Strategies

```python
from meanflow_audio_codec.trainers.loss_strategies import FlowMatchingLoss
from meanflow_audio_codec.trainers.noise_schedules import LinearNoiseSchedule
from meanflow_audio_codec.trainers.time_sampling import LogitNormalTimeSampling
from meanflow_audio_codec.trainers.training_steps import train_step

# Create loss strategy
noise_schedule = LinearNoiseSchedule(noise_min=0.001, noise_max=0.999)
time_sampling = LogitNormalTimeSampling(mean=-0.4, std=1.0)
loss_strategy = FlowMatchingLoss(
    noise_schedule=noise_schedule,
    time_sampling=time_sampling,
    use_weighted_loss=True,
)

# Use in training step
state, loss, key = train_step(state, key, x, loss_strategy)
```

## Noise Schedules

Noise schedules define how to interpolate between data and noise, and how to compute velocity targets.

### Available Noise Schedules

- **LinearNoiseSchedule**: Linear interpolation with configurable min/max noise levels
- **UniformNoiseSchedule**: Uniform interpolation (standard flow matching)

### Creating a Custom Noise Schedule

```python
from meanflow_audio_codec.trainers.noise_schedules import NoiseSchedule
import jax.numpy as jnp

class CustomNoiseSchedule(NoiseSchedule):
    def interpolate(self, x0, x1, t):
        # Your custom interpolation logic
        if t.ndim == 1:
            t = t[:, None]
        # Example: exponential interpolation
        return (1.0 - t) * x0 + jnp.exp(t) * x1
    
    def compute_target(self, x0, x1):
        # Your custom target computation
        return x1 - x0
```

## Time Sampling Strategies

Time sampling strategies define how to sample time values for training.

### Available Time Sampling Strategies

- **UniformTimeSampling**: Uniform sampling from [0, 1]
- **LogitNormalTimeSampling**: Logit-normal distribution (concentrates near 0 and 1)
- **MeanFlowTimeSampling**: Samples (t, r) pairs with r ≤ t for mean flow methods

### Creating a Custom Time Sampling Strategy

```python
from meanflow_audio_codec.trainers.time_sampling import TimeSamplingStrategy
import jax
import jax.numpy as jnp

class CustomTimeSampling(TimeSamplingStrategy):
    def __init__(self, concentration=0.5):
        self.concentration = concentration
    
    def sample_time(self, key, batch_size, dtype=jnp.float32):
        # Your custom time sampling logic
        # Example: beta distribution
        return jax.random.beta(key, self.concentration, self.concentration, (batch_size, 1), dtype=dtype)
```

## Model Factories

Factory functions provide convenient ways to create models from configurations.

### Available Factory Functions

- `create_mlp_flow()`: Create MLP-based flow model
- `create_conv_flow()`: Create ConvNeXt-based flow model
- `create_mlp_mixer_flow()`: Create MLP-Mixer-based flow model
- `create_flow_model(config)`: Generic factory from config

### Using Model Factories

```python
from meanflow_audio_codec.models.factories import create_flow_model, create_mlp_flow
from meanflow_audio_codec.configs.config import TrainFlowConfig

# From config
config = TrainFlowConfig(...)
model = create_flow_model(config)

# Direct factory call
model = create_mlp_flow(
    noise_dimension=784,
    latent_dimension=128,
    num_blocks=4,
    condition_dimension=256,
)
```

## Preprocessing Pipelines

Preprocessing pipelines allow you to compose multiple preprocessing steps.

### Available Pipeline Utilities

- **PreprocessingPipeline**: Composes multiple preprocessing functions
- **Compose**: Composes multiple tokenization strategies
- `create_mdct_pipeline()`: Factory for MDCT-based pipeline
- `create_reshape_pipeline()`: Factory for reshape-based pipeline

### Creating Custom Pipelines

```python
from meanflow_audio_codec.preprocessing.pipelines import PreprocessingPipeline
import jax.numpy as jnp

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def clip(x):
    return jnp.clip(x, -1.0, 1.0)

# Create pipeline
pipeline = PreprocessingPipeline([normalize, clip])

# Apply pipeline
processed = pipeline(x)
```

### Composing Tokenization Strategies

```python
from meanflow_audio_codec.preprocessing.pipelines import Compose
from meanflow_audio_codec.preprocessing.tokenization import MDCTTokenization, ReshapeTokenization

# Compose multiple tokenization strategies
mdct = MDCTTokenization(window_size=512)
reshape = ReshapeTokenization(patch_length=128)

# Note: This is a conceptual example - in practice, you'd typically use one or the other
# pipeline = Compose([mdct, reshape])
```

## Configuration Integration

The configuration system supports composability options:

```python
config = TrainFlowConfig(
    # ... other fields ...
    loss_strategy="flow_matching",  # or "improved_mean_flow"
    noise_schedule="linear",  # or "uniform"
    noise_min=0.001,
    noise_max=0.999,
    time_sampling="logit_normal",  # or "uniform", "mean_flow"
    time_sampling_mean=-0.4,
    time_sampling_std=1.0,
    time_sampling_data_proportion=0.5,  # for mean_flow
    use_weighted_loss=True,
)
```

## Composition Examples

### Example 1: Custom Flow Matching with Uniform Time Sampling

```python
from meanflow_audio_codec.trainers.loss_strategies import FlowMatchingLoss
from meanflow_audio_codec.trainers.noise_schedules import UniformNoiseSchedule
from meanflow_audio_codec.trainers.time_sampling import UniformTimeSampling
from meanflow_audio_codec.trainers.training_steps import train_step

# Create components
noise_schedule = UniformNoiseSchedule()
time_sampling = UniformTimeSampling()
loss_strategy = FlowMatchingLoss(
    noise_schedule=noise_schedule,
    time_sampling=time_sampling,
    use_weighted_loss=False,  # Use MSE instead of weighted L2
)

# Use in training
state, loss, key = train_step(state, key, x, loss_strategy)
```

### Example 2: Mean Flow with Adaptive Reweighting

```python
from meanflow_audio_codec.trainers.loss_strategies import MeanFlowLoss
from meanflow_audio_codec.trainers.noise_schedules import UniformNoiseSchedule
from meanflow_audio_codec.trainers.time_sampling import MeanFlowTimeSampling

# Create with custom parameters
noise_schedule = UniformNoiseSchedule()
time_sampling = MeanFlowTimeSampling(
    mean=-0.5,
    std=0.8,
    data_proportion=0.7,
)
loss_strategy = MeanFlowLoss(
    noise_schedule=noise_schedule,
    time_sampling=time_sampling,
    gamma=0.5,  # Reweighting exponent
    c=1e-3,     # Stability constant
)
```

### Example 3: Improved Mean Flow with Custom Parameters

```python
from meanflow_audio_codec.trainers.loss_strategies import ImprovedMeanFlowLoss
from meanflow_audio_codec.trainers.noise_schedules import LinearNoiseSchedule
from meanflow_audio_codec.trainers.time_sampling import MeanFlowTimeSampling

# Create with custom parameters
noise_schedule = LinearNoiseSchedule(noise_min=0.01, noise_max=0.99)
time_sampling = MeanFlowTimeSampling(
    mean=-0.5,  # More concentration near boundaries
    std=0.8,    # Tighter distribution
    data_proportion=0.7,  # More flow matching boundary conditions
)
loss_strategy = ImprovedMeanFlowLoss(
    noise_schedule=noise_schedule,
    time_sampling=time_sampling,
    use_weighted_loss=True,
)
```

### Example 4: Using Model Factory with Custom Architecture

```python
from meanflow_audio_codec.models.factories import create_mlp_mixer_flow

# Create MLP-Mixer model with custom parameters
model = create_mlp_mixer_flow(
    noise_dimension=784,
    latent_dimension=128,
    num_blocks=6,
    condition_dimension=256,
    token_mix_dim=4096,  # Larger token mixing
    channel_mix_dim=4096,  # Larger channel mixing
    num_channels=32,  # More channels
    num_latent_tokens=64,  # More latent tokens
)
```

## Best Practices

1. **Type Safety**: Always use type hints and ABCs when creating custom components
2. **JAX Compatibility**: Ensure all custom functions are JAX-compatible (pure, jit-friendly)
3. **Backward Compatibility**: When modifying existing components, maintain backward compatibility
4. **Documentation**: Document custom components with clear docstrings
5. **Testing**: Write unit tests for custom components
6. **Performance**: Create strategies once outside the training loop and reuse them for JIT efficiency

## Migration from Old Code

The old training step functions are still available for backward compatibility:

```python
# Old way (still works)
from meanflow_audio_codec.trainers.training_steps import train_step, train_step_improved_mean_flow

if config.use_improved_mean_flow:
    state, loss, key = train_step_improved_mean_flow(state, key, x)
else:
    state, loss, key = train_step(state, key, x)

# New way (recommended)
from meanflow_audio_codec.trainers.training_steps import train_step
from meanflow_audio_codec.trainers.train import create_loss_strategy

loss_strategy = create_loss_strategy(config)
state, loss, key = train_step(state, key, x, loss_strategy)
```

## Further Reading

- [Architecture Guide](architecture.md) - Model architecture details
- [API Reference](api_reference.md) - Complete API documentation
- [Training Guide](../implementation/training/training_session_summary.md) - Training best practices

