# DiT: key equations and contributions

Source: "Scalable Diffusion Models with Transformers" (Peebles, Xie), arXiv: https://arxiv.org/abs/2212.09748

## Notation
- Image: `x` (spatial data)
- Noise: `ϵ ~ N(0, I)` (standard Gaussian)
- Noisy sample: `z_t` at timestep `t`
- Transformer block: `DiTBlock`
- Patch embedding: `p` (patches from input)
- Adaptive layer norm: `AdaLN` (conditioned on timestep and class)

## Background: Diffusion Models

Diffusion models learn to reverse a forward noising process. The forward process adds Gaussian noise:

$$
q(z_t | x) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} x, (1 - \bar{\alpha}_t) I), \tag{1}
$$

where `\bar{\alpha}_t` is a noise schedule. The reverse process is learned:

$$
p_\theta(z_{t-1} | z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t)). \tag{2}
$$

The training objective minimizes the variational lower bound or simplified objective:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,x,ϵ} \|ϵ - ϵ_\theta(z_t, t)\|^2, \tag{3}
$$

where `z_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1 - \bar{\alpha}_t} ϵ` and `ϵ_\theta` predicts the noise.

## DiT Architecture

DiT replaces U-Net backbones with Vision Transformers (ViTs), enabling better scalability.

### Patch Embedding

Input images are divided into patches and embedded:

$$
p = \text{PatchEmbed}(x), \tag{4}
$$

where `x` has shape `(H, W, C)` and `p` has shape `(N, D)` with `N = (H \times W) / P^2` patches of size `P \times P`.

### DiT Blocks

Each DiT block applies:

1. **Self-attention** with adaptive layer norm:
   $$
   h = \text{AdaLN}(p, t, c) + \text{Attention}(\text{AdaLN}(p, t, c)), \tag{5}
   $$

2. **Feed-forward network**:
   $$
   \text{DiTBlock}(p, t, c) = h + \text{FFN}(\text{AdaLN}(h, t, c)), \tag{6}
   $$

where `t` is the timestep embedding and `c` is optional class conditioning.

### Adaptive Layer Norm (AdaLN)

AdaLN modulates layer normalization parameters based on conditioning:

$$
\text{AdaLN}(x, t, c) = \gamma(t, c) \cdot \text{LayerNorm}(x) + \beta(t, c), \tag{7}
$$

where `\gamma` and `\beta` are learned affine parameters from timestep and class embeddings.

### Conditioning Mechanisms

DiT supports three conditioning approaches:

1. **In-context conditioning**: Concatenate timestep/class tokens to sequence
2. **Cross-attention**: Use cross-attention layers for conditioning
3. **Adaptive layer norm (AdaLN)**: Modulate normalization (best performing)

## Key Contributions

1. **Transformer-based diffusion**: Replaces U-Net with Vision Transformer architecture, enabling better scalability with model size.

2. **Scalable architecture**: Demonstrates that transformer architectures scale better than U-Nets for diffusion models, with performance improving monotonically with model size.

3. **Adaptive layer norm**: AdaLN conditioning mechanism outperforms in-context and cross-attention conditioning, providing better integration of timestep and class information.

4. **Patch-based processing**: Operates on image patches rather than pixels, reducing sequence length and enabling efficient attention computation.

5. **Class-conditional generation**: Supports class-conditional image generation with improved quality over unconditional models.

## Architecture Variants

DiT defines several model sizes:

- **DiT-S/2**: Small model (27M parameters)
- **DiT-B/2**: Base model (130M parameters)
- **DiT-L/2**: Large model (458M parameters)
- **DiT-XL/2**: Extra-large model (675M parameters)

The `/2` suffix indicates patch size of 2×2 pixels.

## Training Objective

The training loss combines:

1. **Noise prediction loss**:
   $$
   \mathcal{L}_{\text{noise}} = \mathbb{E}_{t,x,ϵ} \|ϵ - ϵ_\theta(z_t, t, c)\|^2, \tag{8}
   $$

2. **VLB loss** (optional, for improved training):
   $$
   \mathcal{L}_{\text{VLB}} = \mathbb{E}_t D_{KL}(q(z_{t-1} | z_t, x) \| p_\theta(z_{t-1} | z_t)), \tag{9}
   $$

where `c` is class conditioning.

## Sampling

Sampling uses the learned reverse diffusion process:

1. Start from noise: `z_T ~ N(0, I)`
2. Iteratively denoise: `z_{t-1} = \mu_\theta(z_t, t, c) + \sigma_t ϵ`
3. Final sample: `x = z_0`

The number of steps can vary (e.g., 250, 1000 steps) with more steps generally improving quality.

## Performance

**ImageNet 256×256 (class-conditional):**
- **DiT-XL/2**: FID 2.27 (250 steps), 1.73 (1000 steps)
- **DiT-L/2**: FID 3.04 (250 steps)
- **DiT-B/2**: FID 4.60 (250 steps)

**Scalability:**
- Performance improves monotonically with model size
- Transformer architecture enables efficient scaling
- Better parameter efficiency than U-Net baselines

## Relevance to Project

DiT provides architectural inspiration for transformer-based flow models:

1. **Patch embedding**: Can be adapted for MDCT spectral patches or audio tokenization
2. **Adaptive layer norm**: Useful for conditioning flow models on timestep or other metadata
3. **Scalable architecture**: Demonstrates transformer scalability for generative models
4. **Attention mechanisms**: Self-attention can model long-range dependencies in audio sequences

The project uses flow matching rather than diffusion, but DiT's transformer architecture and conditioning mechanisms are relevant for building scalable flow-based models.

## Comparison with Other Methods

1. **vs. U-Net diffusion models**: 
   - Better scalability with model size
   - More efficient attention computation via patches
   - Improved parameter efficiency

2. **vs. SiT**: 
   - DiT focuses on diffusion models
   - SiT extends to flow matching and interpolant-based methods
   - Both use transformer architectures but for different generative frameworks

3. **vs. Vision Transformers (ViTs)**:
   - DiT adapts ViT architecture for generative modeling
   - Adds conditioning mechanisms (AdaLN) for timestep and class
   - Operates in latent space rather than pixel space

## Implementation Details

### Patch Embedding

```python
# Pseudo-code for patch embedding
def patch_embed(x, patch_size=2):
    # x: (B, H, W, C)
    # Extract patches and flatten
    patches = extract_patches(x, patch_size)  # (B, N, P*P*C)
    # Linear projection
    p = Linear(P*P*C, D)(patches)  # (B, N, D)
    return p
```

### DiT Block

```python
# Pseudo-code for DiT block
def dit_block(p, t, c):
    # Adaptive layer norm
    h = adaln(p, t, c) + attention(adaln(p, t, c))
    h = h + ffn(adaln(h, t, c))
    return h
```

### Key Implementation Considerations

1. **Position embeddings**: Add learnable position embeddings to patch sequence
2. **Timestep embedding**: Use sinusoidal or learned embeddings for timestep `t`
3. **Class embedding**: Optional learned embeddings for class conditioning
4. **Attention**: Use efficient attention (e.g., Flash Attention) for large sequences
5. **Normalization**: AdaLN requires separate MLPs for `γ` and `β` parameters

---

## Main Findings

### Key Discoveries

1. **Transformer Scalability**: DiT demonstrates that transformer architectures scale better than U-Nets for diffusion models. Performance improves monotonically with model size (S → B → L → XL), with no performance saturation observed.

2. **Conditioning Mechanism Superiority**: Adaptive Layer Norm (AdaLN) significantly outperforms both in-context conditioning and cross-attention conditioning. AdaLN achieves:
   - Lower FID scores (better quality)
   - More stable training
   - Better integration of timestep and class information

3. **Patch Size Impact**: Smaller patch sizes (2×2) generally yield better results than larger patches (4×4, 8×8) for ImageNet 256×256, balancing sequence length and spatial resolution.

4. **Architecture Efficiency**: Transformer-based diffusion models achieve comparable or better quality than U-Net baselines with:
   - Better parameter efficiency
   - More efficient attention computation via patches
   - Improved scalability to larger models

5. **Class-Conditional Performance**: Class-conditional DiT models significantly outperform unconditional models, demonstrating the effectiveness of AdaLN for conditional generation.

### Empirical Results

- **Scalability**: DiT-XL/2 achieves FID 2.27 with 250 steps, closing the gap with state-of-the-art U-Net models while using a more scalable architecture.
- **Training Stability**: Transformer architecture provides more stable training dynamics compared to U-Net-based diffusion models.
- **Generalization**: The architecture generalizes well across different image resolutions and datasets.

---

## Configurations

### Model Architecture Specifications

**DiT-S/2:**
- Parameters: 27M
- Hidden dimension: 384
- Transformer blocks: 12
- Attention heads: 6
- MLP ratio: 4.0
- Patch size: 2×2

**DiT-B/2:**
- Parameters: 130M
- Hidden dimension: 768
- Transformer blocks: 12
- Attention heads: 12
- MLP ratio: 4.0
- Patch size: 2×2

**DiT-L/2:**
- Parameters: 458M
- Hidden dimension: 1024
- Transformer blocks: 24
- Attention heads: 16
- MLP ratio: 4.0
- Patch size: 2×2

**DiT-XL/2:**
- Parameters: 675M
- Hidden dimension: 1152
- Transformer blocks: 28
- Attention heads: 16
- MLP ratio: 4.0
- Patch size: 2×2

### Training Hyperparameters

**ImageNet 256×256:**
- Optimizer: AdamW
- Learning rate: 1e-4 (base)
- Weight decay: 0.0
- Batch size: 256 (distributed across GPUs)
- Training steps: 7M (approximately 300 epochs)
- Warmup steps: 10,000
- Learning rate schedule: Constant with warmup
- Gradient clipping: None
- Mixed precision: FP16

**Noise Schedule:**
- Type: Linear schedule
- Timesteps: 1000 (training), 250-1000 (sampling)
- Beta start: 0.0001
- Beta end: 0.02

**Data Augmentation:**
- Random horizontal flip: 50% probability
- No other augmentations

### Conditioning Configurations

**AdaLN Implementation:**
- Timestep embedding: Sinusoidal positional embeddings (128-dim)
- Class embedding: Learnable embeddings (768-dim for DiT-B)
- AdaLN MLP: 2-layer MLP with SiLU activation
- Output dimensions: `γ` and `β` each match hidden dimension

**In-Context Conditioning (baseline):**
- Timestep token: Concatenated to patch sequence
- Class token: Concatenated to patch sequence
- Position embeddings: Applied to all tokens

**Cross-Attention Conditioning (baseline):**
- Cross-attention layers: Inserted after self-attention
- Query: From patch embeddings
- Key/Value: From timestep and class embeddings

### Sampling Configurations

**DDPM Sampling:**
- Steps: 250 or 1000
- Sampler: DDPM (denoising diffusion probabilistic model)
- Guidance scale: N/A (class-conditional)

**DDIM Sampling (optional):**
- Steps: 50-250
- Eta parameter: 0.0 (deterministic) or 1.0 (stochastic)

---

## Appendix Content

### Ablation Studies

#### Conditioning Mechanism Comparison

**Results on ImageNet 256×256 (DiT-B/2, 250 steps):**
- AdaLN: FID 4.60
- Cross-attention: FID 5.24
- In-context: FID 5.50

AdaLN provides the best performance with minimal computational overhead.

#### Patch Size Analysis

**DiT-B with different patch sizes (250 steps):**
- 2×2 patches: FID 4.60
- 4×4 patches: FID 5.11
- 8×8 patches: FID 5.67

Smaller patches provide better quality but increase sequence length and memory requirements.

#### Depth and Width Scaling

**Scaling analysis:**
- Increasing depth (blocks): Consistent improvement
- Increasing width (hidden dim): Consistent improvement
- Optimal ratio: Depth and width scale together for best efficiency

#### Position Embedding Analysis

- Learnable vs. Sinusoidal: Learnable performs slightly better
- 2D vs. 1D: 2D position embeddings (row/column) provide marginal improvement

### Additional Experimental Results

#### Unconditional Generation

**ImageNet 256×256 (unconditional):**
- DiT-XL/2: FID 3.04 (250 steps)
- DiT-L/2: FID 3.94 (250 steps)
- DiT-B/2: FID 5.25 (250 steps)

Unconditional models perform worse than class-conditional, as expected.

#### Different Image Resolutions

**CIFAR-10 (32×32):**
- DiT-XL/2: FID 3.04
- Demonstrates architecture flexibility across resolutions

#### Computational Efficiency

**Training time (ImageNet 256×256, DiT-XL/2):**
- Training steps: 7M
- GPU hours: ~10,000 A100 GPU hours
- Throughput: ~2.5 samples/sec per GPU

**Inference time (250 steps):**
- DiT-XL/2: ~1.3 seconds per image (A100)
- DiT-B/2: ~0.4 seconds per image (A100)

### Implementation Details

#### Attention Optimization

- **Flash Attention**: Used for memory-efficient attention computation
- **Gradient checkpointing**: Applied to reduce memory usage during training
- **Mixed precision**: FP16 training for faster computation

#### Codebook and Quantization (if applicable)

- Not applicable to DiT (no quantization in base model)

#### Training Stability Techniques

1. **Gradient accumulation**: Used for effective larger batch sizes
2. **EMA (Exponential Moving Average)**: Optional, for model checkpointing
3. **Learning rate warmup**: 10,000 steps to stabilize early training

### Mathematical Derivations

#### AdaLN Formulation

The adaptive layer norm computes:

$$
\text{AdaLN}(x, t, c) = \gamma(t, c) \odot \text{LayerNorm}(x) + \beta(t, c),
$$

where:
- `\gamma(t, c) = \text{MLP}_\gamma(\text{Embed}(t) \| \text{Embed}(c))`
- `\beta(t, c) = \text{MLP}_\beta(\text{Embed}(t) \| \text{Embed}(c))`
- `\|` denotes concatenation
- `\odot` denotes element-wise multiplication

#### Diffusion Process Integration

The transformer processes noisy images `z_t` at timestep `t`:

$$
z_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon,
$$

where `\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)` and `\beta_s` is the noise schedule.

### Additional Visualizations

- **Attention maps**: Visualizations of self-attention patterns in DiT blocks
- **Generated samples**: High-quality class-conditional ImageNet samples
- **Training curves**: Loss and FID progression during training
- **Architecture diagrams**: Detailed transformer block structure

### Comparison with Baselines

#### vs. U-Net Diffusion Models

**ADM (U-Net baseline) on ImageNet 256×256:**
- ADM: FID 2.07 (250 steps)
- DiT-XL/2: FID 2.27 (250 steps)
- DiT achieves competitive performance with better scalability

#### vs. Other Transformer Diffusion Models

- **GenViT**: DiT outperforms in class-conditional generation
- **Diffusion Transformer**: DiT's AdaLN provides better conditioning

### Failure Cases and Limitations

1. **High-resolution images**: Performance degrades for 512×512+ without architectural modifications
2. **Long sequences**: Very long sequences (small patches, large images) require significant memory
3. **Training time**: Large models require extensive computational resources

### Future Directions

1. **Efficient attention**: Further optimization for longer sequences
2. **Multi-resolution training**: Training on multiple resolutions simultaneously
3. **Conditional generation**: Extension to text-to-image and other conditioning modalities
4. **Fewer sampling steps**: Integration with consistency models or distillation techniques

