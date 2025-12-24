# Experimental Tables and Ablations for Final Paper

This document outlines the tables that would appear in a typical Kaiming He-style paper and the ablations/configurations needed to populate them.

## Research Questions (from README)

The project systematically studies:
1. **Method Progression**: Autoencoder → Flow Matching → Mean Flow → Improved Mean Flow
2. **Architecture Progression**: MLP → MLP Mixer → ConvNet
3. **Tokenization Strategy**: MDCT-based vs reshape-based
4. **Dataset Transfer**: MNIST → Audio

## Expected Tables in Final Paper

### Table 1: Main Results - Comprehensive Method Comparison

**Purpose**: Show performance across all methods, architectures, and datasets.

**Structure**:
| Method | Architecture | Dataset | Tokenization | FID ↓ | KID ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | NFE | Training Time | Params |
|--------|-------------|---------|--------------|-------|-------|-------|--------|--------|-----|---------------|--------|
| Autoencoder | MLP | MNIST | Reshape | ... | ... | ... | ... | ... | 1 | ... | ... |
| Autoencoder | MLP | MNIST | MDCT | ... | ... | ... | ... | ... | 1 | ... | ... |
| Flow Matching | MLP | MNIST | Reshape | ... | ... | ... | ... | ... | 250 | ... | ... |
| Flow Matching | MLP | MNIST | MDCT | ... | ... | ... | ... | ... | 250 | ... | ... |
| Mean Flow | MLP | MNIST | Reshape | ... | ... | ... | ... | ... | 1 | ... | ... |
| Mean Flow | MLP | MNIST | MDCT | ... | ... | ... | ... | ... | 1 | ... | ... |
| Improved Mean Flow | MLP | MNIST | Reshape | ... | ... | ... | ... | ... | 1 | ... | ... |
| Improved Mean Flow | MLP | MNIST | MDCT | ... | ... | ... | ... | ... | 1 | ... | ... |
| ... (repeat for MLP Mixer, ConvNet) | | | | | | | | | | | |
| ... (repeat for Audio dataset) | | | | | | | | | | | |

**Metrics Needed**:
- **FID** (Fréchet Inception Distance): Already implemented
- **KID** (Kernel Inception Distance): Already implemented
- **MSE** (Mean Squared Error): Need to track during evaluation
- **PSNR** (Peak Signal-to-Noise Ratio): Need to implement
- **SSIM** (Structural Similarity Index): Need to implement (for images/audio)
- **NFE** (Number of Function Evaluations): Track sampling steps
- **Training Time**: Track wall-clock time
- **Parameters**: Track model size

**Configs Needed**: All combinations of:
- 4 methods × 3 architectures × 2 tokenizations × 2 datasets = 48 configurations

**Status**: Configs exist for most combinations, but need to ensure:
- Tokenization strategy is configurable (MDCT vs reshape)
- All metrics are computed and logged
- Training time and parameter counts are tracked

---

### Table 2: Ablation Study - Method Components

**Purpose**: Isolate the contribution of each method component.

**Structure**:
| Method Variant | Description | FID ↓ | KID ↓ | Training Stability |
|----------------|-------------|-------|-------|-------------------|
| Autoencoder (baseline) | Reconstruction only | ... | ... | ... |
| Flow Matching | Standard flow matching | ... | ... | ... |
| Mean Flow | Average velocity (1-step) | ... | ... | ... |
| Improved Mean Flow | iMF with improved dynamics | ... | ... | ... |
| Mean Flow (w/o stop-grad) | Ablation: no stop-gradient | ... | ... | ... |
| Improved Mean Flow (w/o gamma) | Ablation: no gamma parameter | ... | ... | ... |
| Improved Mean Flow (w/o flow_ratio) | Ablation: no flow_ratio | ... | ... | ... |

**Ablations Needed**:
1. **Stop-gradient ablation**: Mean Flow with/without stop-gradient on target
2. **Gamma parameter**: Improved Mean Flow with different gamma values (0.0, 0.5, 1.0, 2.0)
3. **Flow ratio**: Improved Mean Flow with different flow_ratio values (0.0, 0.5, 1.0)
4. **Time sampling**: Uniform vs importance sampling for `t` and `r`
5. **Loss weighting**: Uniform vs time-dependent weighting

**Configs Needed**: 
- Base configs for each method variant
- Hyperparameter sweep configs for gamma, flow_ratio
- Configs with different time sampling strategies
- Configs with different loss weighting schemes

---

### Table 3: Architecture Ablation Study

**Purpose**: Understand impact of architectural choices.

**Structure**:
| Architecture | Hidden Dim | Num Blocks | Attention Heads | Params | FID ↓ | Training Speed | Memory |
|--------------|------------|------------|-----------------|--------|-------|----------------|--------|
| MLP (small) | 256 | 4 | N/A | ... | ... | ... | ... |
| MLP (medium) | 512 | 8 | N/A | ... | ... | ... | ... |
| MLP (large) | 1024 | 16 | N/A | ... | ... | ... | ... |
| MLP Mixer (small) | 256 | 4 | N/A | ... | ... | ... | ... |
| MLP Mixer (medium) | 512 | 8 | N/A | ... | ... | ... | ... |
| MLP Mixer (large) | 1024 | 16 | N/A | ... | ... | ... | ... |
| ConvNet (small) | 256 | 4 | N/A | ... | ... | ... | ... |
| ConvNet (medium) | 512 | 8 | N/A | ... | ... | ... | ... |
| ConvNet (large) | 1024 | 16 | N/A | ... | ... | ... | ... |

**Ablations Needed**:
1. **Model size scaling**: Small/medium/large variants of each architecture
2. **Depth scaling**: Vary number of blocks (4, 8, 12, 16, 24)
3. **Width scaling**: Vary hidden dimension (256, 512, 768, 1024, 1152)
4. **Attention heads** (for transformer-based): 6, 8, 12, 16
5. **MLP ratio** (for transformer-based): 2.0, 4.0, 8.0

**Configs Needed**:
- Configs with varying `num_blocks` (4, 8, 12, 16, 24)
- Configs with varying `latent_dimension` / `condition_dimension` (256, 512, 768, 1024)
- Configs for transformer variants (if applicable) with attention heads and MLP ratio

---

### Table 4: Tokenization Strategy Comparison

**Purpose**: Compare MDCT vs reshape-based tokenization.

**Structure**:
| Method | Architecture | Tokenization | FID ↓ | Reconstruction Quality | Training Stability | Token Efficiency |
|--------|-------------|--------------|-------|------------------------|-------------------|------------------|
| Improved Mean Flow | MLP | Reshape | ... | ... | ... | ... |
| Improved Mean Flow | MLP | MDCT | ... | ... | ... | ... |
| Improved Mean Flow | MLP Mixer | Reshape | ... | ... | ... | ... |
| Improved Mean Flow | MLP Mixer | MDCT | ... | ... | ... | ... |
| Improved Mean Flow | ConvNet | Reshape | ... | ... | ... | ... |
| Improved Mean Flow | ConvNet | MDCT | ... | ... | ... | ... |

**Ablations Needed**:
1. **MDCT window size**: Different MDCT window sizes (512, 1024, 2048)
2. **MDCT hop size**: Different hop sizes (50%, 75% overlap)
3. **Reshape patch size**: Different patch sizes for reshape tokenization
4. **Token dimension**: Impact of token embedding dimension

**Configs Needed**:
- Configs with `tokenization="reshape"` vs `tokenization="mdct"`
- Configs with different MDCT parameters (window_size, hop_size)
- Configs with different patch sizes for reshape

---

### Table 5: Hyperparameter Sensitivity

**Purpose**: Show robustness to hyperparameter choices.

**Structure**:
| Hyperparameter | Value Range | Best Value | FID (min) | FID (max) | Sensitivity |
|----------------|-------------|------------|-----------|-----------|-------------|
| Learning Rate | [1e-5, 5e-4] | ... | ... | ... | Low/Medium/High |
| Weight Decay | [0, 1e-3] | ... | ... | ... | ... |
| Batch Size | [32, 256] | ... | ... | ... | ... |
| Gamma (iMF) | [0.0, 2.0] | ... | ... | ... | ... |
| Flow Ratio (iMF) | [0.0, 1.0] | ... | ... | ... | ... |
| Sample Steps | [1, 250] | ... | ... | ... | ... |

**Ablations Needed**:
1. **Learning rate sweep**: 1e-5, 5e-5, 1e-4, 5e-4
2. **Weight decay sweep**: 0, 1e-6, 1e-4, 1e-3
3. **Batch size sweep**: 32, 64, 128, 256 (within memory constraints)
4. **Gamma sweep**: 0.0, 0.5, 1.0, 1.5, 2.0
5. **Flow ratio sweep**: 0.0, 0.25, 0.5, 0.75, 1.0
6. **Sample steps sweep**: 1, 10, 50, 100, 250

**Configs Needed**:
- Grid search configs for each hyperparameter
- Or configs with hyperparameter ranges for automated sweeps

---

### Table 6: Computational Efficiency

**Purpose**: Compare speed, memory, and efficiency trade-offs.

**Structure**:
| Method | Architecture | NFE | Inference Time (ms) | Training Time (hrs) | Memory (GB) | FID ↓ | Speedup vs Flow Matching |
|--------|-------------|-----|---------------------|---------------------|-------------|-------|--------------------------|
| Flow Matching | MLP | 250 | ... | ... | ... | ... | 1.0× |
| Mean Flow | MLP | 1 | ... | ... | ... | ... | 250× |
| Improved Mean Flow | MLP | 1 | ... | ... | ... | ... | 250× |
| Flow Matching | ConvNet | 250 | ... | ... | ... | ... | 1.0× |
| Mean Flow | ConvNet | 1 | ... | ... | ... | ... | 250× |
| Improved Mean Flow | ConvNet | 1 | ... | ... | ... | ... | 250× |

**Metrics Needed**:
- **Inference time**: Time per sample (single batch)
- **Training time**: Total wall-clock training time
- **Memory usage**: Peak GPU/RAM memory during training and inference
- **Throughput**: Samples per second
- **Speedup**: Relative to Flow Matching baseline

**Configs Needed**:
- Same as main results, but with profiling enabled
- Configs for benchmarking inference speed
- Configs for memory profiling

---

### Table 7: Dataset Transfer (MNIST → Audio)

**Purpose**: Show generalization from MNIST to audio domain.

**Structure**:
| Method | Architecture | Dataset | FID ↓ | Domain-Specific Metric | Transfer Success |
|--------|-------------|---------|-------|------------------------|------------------|
| Improved Mean Flow | MLP | MNIST | ... | ... | Baseline |
| Improved Mean Flow | MLP | Audio | ... | ... | ... |
| Improved Mean Flow | MLP Mixer | MNIST | ... | ... | Baseline |
| Improved Mean Flow | MLP Mixer | Audio | ... | ... | ... |
| Improved Mean Flow | ConvNet | MNIST | ... | ... | Baseline |
| Improved Mean Flow | ConvNet | Audio | ... | ... | ... |

**Metrics Needed**:
- **Audio-specific metrics**: 
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - MOS (Mean Opinion Score) - if available
  - Spectral distance metrics
- **Domain transfer metrics**: How well MNIST performance predicts audio performance

**Configs Needed**:
- Configs for audio dataset (already exist)
- Configs with audio-specific evaluation metrics
- Configs for cross-domain evaluation

---

### Table 8: Comparison with Baselines

**Purpose**: Compare against published baselines and reference implementations.

**Structure**:
| Method | Architecture | Dataset | Our FID | Baseline FID | Reference | Improvement |
|--------|-------------|---------|---------|--------------|-----------|-------------|
| Flow Matching | ConvNet | MNIST | ... | ... | Lipman et al. | ... |
| Mean Flow | ConvNet | MNIST | ... | ... | Geng et al. (2025) | ... |
| Improved Mean Flow | ConvNet | MNIST | ... | ... | Geng et al. (2024) | ... |
| Improved Mean Flow | ConvNet | Audio | ... | ... | MDCTCodec | ... |

**Baselines Needed**:
- **Flow Matching**: Reference from Lipman et al. (2023)
- **Mean Flow**: Reference from Geng et al. (2025)
- **Improved Mean Flow**: Reference from Geng et al. (2024)
- **MDCTCodec**: For audio codec comparison
- **EnCodec**: For audio codec comparison (if applicable)

**Configs Needed**:
- Configs matching published baseline settings exactly
- Configs for reproducing baseline results

---

## Additional Ablations and Configs Needed

### 1. Training Dynamics Ablations

**Configs needed for**:
- **Learning rate schedule**: Constant, cosine, linear decay, warmup variants
- **Optimizer**: Adam, AdamW, SGD with momentum
- **Gradient clipping**: None, value clipping, norm clipping
- **Mixed precision**: FP32, FP16, BF16
- **EMA (Exponential Moving Average)**: With/without EMA for model weights

### 2. Sampling Ablations

**Configs needed for**:
- **ODE solver**: Euler, Heun, RK4, adaptive solvers
- **Number of steps**: 1, 10, 50, 100, 250
- **Guidance scale** (for conditional models): 1.0, 1.5, 2.0, 3.0
- **Temperature scaling**: Different noise temperatures

### 3. Conditioning Ablations

**Configs needed for**:
- **Class conditioning**: With/without class labels
- **AdaLN variants**: Different adaptive normalization strategies
- **Conditioning dropout**: 0%, 10%, 20% for classifier-free guidance

### 4. Loss Function Ablations

**Configs needed for**:
- **Loss type**: L2, L1, Huber, normalized MSE
- **Time-dependent weighting**: Uniform, `1/t`, `t(1-t)`, learned weighting
- **Multi-scale loss**: Single scale vs multi-scale

### 5. Data Augmentation Ablations

**Configs needed for**:
- **Augmentation strategies**: None, random crop, random flip, mixup, etc.
- **Augmentation strength**: Different augmentation probabilities

---

## Implementation Checklist

### Metrics to Implement
- [ ] PSNR (Peak Signal-to-Noise Ratio)
- [ ] SSIM (Structural Similarity Index)
- [ ] PESQ (for audio)
- [ ] STOI (for audio)
- [ ] Spectral distance metrics (for audio)
- [ ] Training time tracking
- [ ] Inference time tracking
- [ ] Memory usage tracking
- [ ] Parameter counting

### Configurations to Create
- [ ] Tokenization strategy configs (MDCT vs reshape)
- [ ] Hyperparameter sweep configs (gamma, flow_ratio, lr, etc.)
- [ ] Architecture scaling configs (small/medium/large)
- [ ] Ablation configs (stop-gradient, loss weighting, etc.)
- [ ] Baseline reproduction configs
- [ ] Profiling/benchmarking configs

### Evaluation Pipeline Enhancements
- [ ] Automated metric computation and logging
- [ ] Table generation from results
- [ ] Statistical significance testing
- [ ] Error bars and confidence intervals
- [ ] Visualization of results

### Training Infrastructure
- [ ] Hyperparameter sweep framework
- [ ] Distributed training support (if needed)
- [ ] Checkpointing and resuming
- [ ] Experiment tracking (weights & biases, tensorboard, etc.)

---

## Notes

1. **Resource Constraints**: Given M1 MacBook with 16GB RAM, some large-scale experiments may need to be run on cloud infrastructure or scaled down appropriately.

2. **Prioritization**: Focus on:
   - Main results table (Table 1) - highest priority
   - Method ablation (Table 2) - core contribution
   - Tokenization comparison (Table 4) - key research question
   - Computational efficiency (Table 6) - practical importance

3. **Baseline Comparisons**: Some baseline comparisons may require reproducing results from papers, which may not be feasible. Focus on internal comparisons and reference implementations.

4. **Audio Metrics**: Audio-specific metrics (PESQ, STOI) may require additional dependencies and may not be applicable to all audio datasets.

