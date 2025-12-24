# Training Session Summary: ConditionalConvFlow on MNIST

**Date**: December 24, 2024  
**Status**: Training completed successfully, but model not learning (loss stuck at ~1.0)

## Overview

Successfully set up and ran training for `ConditionalConvFlow` model on MNIST dataset using JAX Metal backend on Apple M1. The training infrastructure is working correctly, but the model requires further tuning to learn effectively.

## Key Achievements

### 1. Metal Backend Compatibility ✅
- Fixed Metal incompatibility issues:
  - **GlobalResponseNormalization**: Replaced `jnp.linalg.norm(x, ord=2, ...)` with manual L2 norm computation (`jnp.sqrt(jnp.sum(x * x, ...))`) to avoid `eigh` primitive not supported on Metal
  - **ConvNeXtBlock**: Removed `feature_group_count` (depthwise convolutions) as they're not supported on Metal; converted to regular convolutions
- Model now runs successfully on Metal GPU

### 2. Model Size Reduction ✅
- Reduced model from **53M → 7.26M parameters** to meet <10M target:
  - Kernel size: `(7,7)` → `(3,3)`
  - Base channels: `64` → `16`
  - Condition dimension: `128` → `32`
  - Latent dimension: `256` → `32`
  - Expansion ratio: `4x` → `2x`
  - Added 128-dim bottleneck layers in conditional projections
- Checkpoint size reduced from **3.5GB → 83MB**

### 3. Data Loading Reliability ✅
- Implemented `load_data_simple()` to pre-load MNIST samples into memory
- Bypassed TensorFlow Dataset iterator issues that caused crashes on Metal
- Reduced initial data load from 20,000+ to 5,000-10,000 samples for faster startup

### 4. Training Infrastructure ✅
- Created robust training script: `train_conv_flow_mnist.py`
- Implemented checkpointing with resumption support
- Added JSONL logging (`train_log.jsonl`)
- Created helper scripts:
  - `run_training.sh` - Start training in background
  - `monitor_training.sh` - Monitor training progress
  - `kill_training.sh` - Stop training processes

## Training Results

### Completed Run
- **Total steps**: 2000 (completed)
- **Final loss**: 0.999997 (not learning)
- **Checkpoints**: 83MB each, saved every 200 steps
- **Samples**: Generated at steps 1200, 1400, 1600, 1800, 2000
- **Training time**: ~2 hours (estimated)

### Output Files
- **Checkpoints**: `outputs/trial_conv_flow/checkpoints/`
  - `step_02000.msgpack` (final)
  - `step_01800.msgpack`
  - `latest.msgpack`
- **Samples**: `outputs/trial_conv_flow/samples/`
  - Multiple PNG files at various training steps
- **Logs**: `outputs/trial_conv_flow/logs/`
  - `train_log.jsonl` - JSONL training metrics
  - `training.log` - Combined stdout/stderr
  - `stdout.log`, `stderr.log` - Separate logs

## Current Issue: Model Not Learning

### Symptoms
- Loss stuck at ~0.999997 throughout training
- No improvement over 2000 steps
- Generated samples likely show no meaningful structure

### Potential Causes
1. **Learning rate too low**: Current `base_lr=2e-4` may be insufficient
2. **Model capacity too small**: 7.26M parameters may be insufficient for the task
3. **Training objective**: The Improved Mean Flow loss computation may need adjustment
4. **Data preprocessing**: Normalization or preprocessing may be incorrect
5. **Gradient flow**: Gradients may be vanishing or exploding

### Next Steps for Investigation

1. **Check gradient magnitudes**:
   ```python
   # Add gradient norm logging to training loop
   grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
   ```

2. **Verify loss computation**:
   - Check if `weighted_l2_loss` is computing correctly
   - Verify `train_step_improved_mean_flow` logic matches paper
   - Check if target values are reasonable

3. **Experiment with hyperparameters**:
   - Increase learning rate: `2e-4` → `5e-4` or `1e-3`
   - Increase model capacity slightly (e.g., `condition_dim=64`, `latent_dim=64`)
   - Adjust batch size: `32` → `64` or `16`

4. **Compare with baseline**:
   - Try standard Flow Matching (`train_step` instead of `train_step_improved_mean_flow`)
   - Compare loss values between methods

5. **Inspect generated samples**:
   - Check if samples show any structure
   - Verify sampling process is correct

## File Structure

### Main Training Script
- **`train_conv_flow_mnist.py`**: Main training script with checkpointing, logging, and Metal compatibility

### Helper Scripts
- **`run_training.sh`**: Start training in background with PID management
- **`monitor_training.sh`**: Monitor training progress and status
- **`kill_training.sh`**: Stop training processes and clean up

### Model Files
- **`meanflow_audio_codec/models/conv_flow.py`**: 
  - `ConditionalConvFlow` - Main model
  - `ConditionalConvNeXtBlock` - Conditional block with bottleneck
  - `ConvNeXtBlock` - Standard block (Metal-compatible)
  - `GlobalResponseNormalization` - GRN layer (Metal-compatible)

### Training Components
- **`meanflow_audio_codec/trainers/training_steps.py`**: 
  - `train_step_improved_mean_flow` - Improved Mean Flow training step
  - `train_step` - Baseline Flow Matching step
- **`meanflow_audio_codec/trainers/utils.py`**: 
  - `LogWriter`, `save_checkpoint`, `load_checkpoint_and_resume`, `plot_samples`

## Configuration

### Current Training Config
```python
batch_size = 32
n_steps = 2000
sample_every = 200
sample_steps = 50
base_lr = 2e-4
weight_decay = 1e-4
seed = 42
```

### Model Config
```python
noise_dimension = 784  # 28*28 for MNIST
condition_dimension = 32
latent_dimension = 32
num_blocks = 4
num_classes = 10
image_size = 28
use_grn = True
```

## Metal Backend Notes

- JAX Metal is experimental and has limitations
- Some operations not supported:
  - `jnp.linalg.norm` with `ord=2` (use manual L2 norm)
  - Depthwise convolutions with `feature_group_count`
  - Complex number operations
- Platform detection: `jax.default_backend() == 'METAL'`
- Warning: "Platform 'METAL' is experimental" is expected

## Commands Reference

### Start Training
```bash
./run_training.sh
```

### Monitor Training
```bash
./monitor_training.sh
# or
tail -f outputs/trial_conv_flow/logs/training.log
```

### Stop Training
```bash
./kill_training.sh
```

### Check Training Progress
```bash
# View JSONL log
tail -f outputs/trial_conv_flow/logs/train_log.jsonl | jq

# Count steps
wc -l outputs/trial_conv_flow/logs/train_log.jsonl
```

## Technical Details

### Metal Compatibility Fixes Applied

1. **GlobalResponseNormalization**:
   ```python
   # Before (not Metal-compatible):
   norm = jnp.linalg.norm(x, ord=2, axis=spatial_dims, keepdims=True)
   
   # After (Metal-compatible):
   norm = jnp.sqrt(jnp.sum(x * x, axis=spatial_dims, keepdims=True))
   ```

2. **ConvNeXtBlock**:
   ```python
   # Before (depthwise conv, not Metal-compatible):
   nn.Conv(features=self.dim, kernel_size=(7,7), feature_group_count=self.dim)
   
   # After (regular conv, Metal-compatible):
   nn.Conv(features=self.dim // 4, kernel_size=(3,3))  # No feature_group_count
   ```

### Data Loading Strategy

- Pre-loads fixed number of samples into memory as NumPy arrays
- Creates manual batch iterator to avoid TFDS iterator issues
- Reduces startup time by limiting initial data load

## References

- Improved Mean Flow paper: "Improved Mean Flows: On the Challenges of Fastforward Generative Models" (Geng et al., 2024)
- Documentation: `documentation/improved_meanflow/improved_meanflow_key_eqn.md`
- Model architecture: `meanflow_audio_codec/models/conv_flow.py`
- Training step: `meanflow_audio_codec/trainers/training_steps.py`

