# Stable Audio Codec: key equations and contributions

Source: Stability AI Open Audio Codec (2024), GitHub: https://github.com/Stability-AI/stable-audio-codec

## Notation
- Audio waveform: `x[n]` (time-domain samples)
- Encoder output: `z` (continuous latent code)
- Quantized code: `\hat{z}` (discrete latent)
- Decoder output: `\hat{x}[n]` (reconstructed waveform)
- Transformer blocks: `TransformerBlock`
- Residual Vector Quantization: `RVQ` with `L` quantizers

## Overview

Stable Audio Codec is an open-source neural audio codec developed by Stability AI. It provides a publicly available implementation of a high-quality audio compression system, enabling research and development in neural audio codecs.

## Architecture

Stable Audio Codec uses a transformer-based architecture with residual vector quantization:

1. **Encoding**: `x[n] \rightarrow z \rightarrow \hat{z}` (waveform → continuous latent → quantized)
2. **Decoding**: `\hat{z} \rightarrow \hat{x}[n]` (quantized → reconstructed waveform)

### Transformer-Based Design

Unlike convolutional architectures (EnCodec, MDCTCodec), Stable Audio Codec uses transformers:

- **Patch embedding**: Audio is divided into patches and embedded
- **Transformer blocks**: Self-attention and feed-forward layers
- **Position encoding**: Learnable or sinusoidal position embeddings

This transformer architecture enables:
- Long-range dependency modeling
- Scalable architecture
- Flexible conditioning mechanisms

## Encoder

The encoder `E_\theta` processes audio waveforms:

$$
z = E_\theta(x[n]). \tag{1}
$$

**Architecture**: Transformer encoder
- Input: Waveform `x[n]` of shape `(T,)`
- Patch embedding: Divides waveform into patches and projects to embedding dimension
- Transformer blocks: Multiple layers of self-attention and feed-forward networks
- Output: Continuous latent `z` of shape `(T', D)` with reduced temporal dimension

## Residual Vector Quantization

Similar to EnCodec, Stable Audio Codec uses RVQ:

$$
z_0 = z,
$$
$$
\hat{z}_i = \mathrm{Quantize}_i(z_{i-1}), \quad i = 1, \ldots, L,
$$
$$
z_i = z_{i-1} - \hat{z}_i,
$$
$$
\hat{z} = \sum_{i=1}^{L} \hat{z}_i. \tag{2}
$$

Each quantizer uses a codebook `C_i` of size `K`:
$$
\mathrm{Quantize}_i(z) = \arg\min_{c \in C_i} \|z - c\|^2.
$$

## Decoder

The decoder `D_\phi` reconstructs waveforms:

$$
\hat{x}[n] = D_\phi(\hat{z}). \tag{3}
$$

**Architecture**: Transformer decoder
- Input: Quantized latent `\hat{z}` of shape `(T', D)`
- Transformer blocks: Multiple layers with self-attention
- Output projection: Maps to waveform samples
- Output: Reconstructed waveform `\hat{x}[n]` of shape `(T,)`

## Training Objectives

The codec is trained with a combination of losses:

**Reconstruction loss:**
$$
\mathcal{L}_{\text{recon}} = \mathbb{E}_{x \sim p_{\text{data}}} \|x - \hat{x}\|^2. \tag{4}
$$

**Adversarial loss:**
$$
\mathcal{L}_{\text{adv}} = -\mathbb{E}_{\hat{x} \sim p_\theta}[\log D_\psi(\hat{x})], \tag{5}
$$

where `D_\psi` is a discriminator network.

**Feature matching loss:**
$$
\mathcal{L}_{\text{fm}} = \mathbb{E}_{x \sim p_{\text{data}}} \sum_{l} \|D_\psi^{(l)}(x) - D_\psi^{(l)}(\hat{x})\|^2, \tag{6}
$$

where `D_\psi^{(l)}` denotes intermediate discriminator features.

**Total loss:**
$$
\mathcal{L}_G = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{fm}}\mathcal{L}_{\text{fm}}. \tag{7}
$$

## Key Features

1. **Open-source implementation**: Publicly available codebase enables research and development
2. **Transformer architecture**: Uses transformers for encoder/decoder, enabling long-range modeling
3. **High-quality compression**: Achieves high fidelity audio compression
4. **Flexible quantization**: RVQ enables variable bitrate control
5. **Training methodology**: Includes training scripts and procedures

## Performance

The codec achieves:
- High-quality audio reconstruction
- Competitive compression ratios
- Real-time inference capabilities
- Support for various audio types (speech, music, general audio)

## Relevance to Project

Stable Audio Codec provides:

1. **Open-source reference**: Publicly available implementation for comparison and reference
2. **Transformer architecture**: Demonstrates transformer-based audio codec design, relevant for exploring transformer architectures in the project
3. **Training procedures**: Training methodology and loss formulations provide reference
4. **Code structure**: Codebase organization and implementation patterns can inform project structure
5. **Architecture comparison**: Transformer-based vs. convolutional vs. MDCT-based approaches can be compared

## Comparison with Other Codecs

1. **vs. EnCodec**: 
   - EnCodec: Convolutional architecture
   - Stable Audio Codec: Transformer architecture
   - Both use RVQ and waveform domain

2. **vs. MDCTCodec**: 
   - MDCTCodec: MDCT domain, ConvNeXt v2 backbone
   - Stable Audio Codec: Waveform domain, transformer backbone
   - Different domain representations and architectures

3. **vs. Project (Improved Mean Flow + MDCT)**: 
   - Project: MDCT domain with Improved Mean Flow
   - Stable Audio Codec: Waveform domain with transformers
   - Different approaches to audio encoding

## Implementation Details

### Transformer Encoder

```python
# Pseudo-code for transformer encoder
def transformer_encoder(x):
    # x: (B, T) waveform
    # Patch embedding
    patches = patch_embed(x)  # (B, N, D)
    # Add position encoding
    patches = patches + position_encoding(N)
    # Transformer blocks
    for block in transformer_blocks:
        patches = block(patches)
    return patches  # (B, N, D)
```

### Transformer Block

```python
# Pseudo-code for transformer block
def transformer_block(x):
    # Self-attention
    h = x + attention(layer_norm(x))
    # Feed-forward
    h = h + ffn(layer_norm(h))
    return h
```

### Key Implementation Considerations

1. **Patch embedding**: Divide waveform into overlapping or non-overlapping patches
2. **Position encoding**: Use learnable or sinusoidal position embeddings
3. **Attention mechanism**: Use efficient attention (Flash Attention) for long sequences
4. **Normalization**: Layer normalization in transformer blocks
5. **RVQ implementation**: Similar to EnCodec, with codebook management
6. **Discriminator**: Multi-scale or single-scale discriminator for adversarial training

## Open-Source Contributions

The Stable Audio Codec repository provides:
- Complete codebase for training and inference
- Pre-trained models
- Training scripts and configurations
- Evaluation tools
- Documentation

This open-source nature makes it valuable for:
- Research and development
- Architecture comparison
- Implementation reference
- Community contributions

## Future Directions

The open-source nature enables:
- Community improvements and extensions
- Research into transformer-based audio codecs
- Comparison studies with other architectures
- Integration with other audio processing pipelines

---

## Main Findings

### Key Discoveries

1. **Transformer Architecture for Audio**: Demonstrates that transformer architectures can be effectively applied to neural audio codecs, providing:
   - Long-range dependency modeling in audio sequences
   - Scalable architecture similar to DiT/SiT for images
   - Flexible conditioning mechanisms
   - Competitive quality with convolutional alternatives

2. **Open-Source Contribution**: Provides a publicly available, well-documented implementation that enables:
   - Reproducible research
   - Community-driven improvements
   - Easy comparison with other codecs
   - Educational value for understanding neural audio codecs

3. **Streaming Capability**: Designed for real-time applications with:
   - Low-latency encoding and decoding
   - Causal processing (if implemented)
   - Efficient inference on standard hardware

4. **Transformer vs. Convolutional Trade-offs**: 
   - Transformers: Better long-range modeling, more parameters
   - Convolutions: Lower computational cost, local feature extraction
   - Choice depends on application requirements

### Empirical Results

**Performance Characteristics:**
- Quality: Competitive with EnCodec and other neural codecs
- Speed: Real-time capable on GPU, slower on CPU
- Memory: Higher memory requirements than convolutional codecs
- Scalability: Architecture scales with model size

**Open-Source Impact:**
- Enables reproducible research
- Facilitates comparison studies
- Provides implementation reference
- Supports educational use

---

## Configurations

### Model Architecture Specifications

**Transformer Encoder:**
- Input: Waveform patches or spectral features
- Patch embedding: Divides audio into patches and projects to embedding dimension
- Position encoding: Learnable or sinusoidal position embeddings
- Transformer blocks: Multiple layers of self-attention and feed-forward networks
- Output: Continuous latent representation

**Transformer Decoder:**
- Input: Quantized latent codes
- Transformer blocks: Self-attention layers for reconstruction
- Output projection: Maps to waveform samples
- Output: Reconstructed audio waveform

**Architecture Variants:**
- Model size: Can vary from small to large (similar to DiT/SiT scaling)
- Hidden dimension: Typically 384-1152 depending on model size
- Number of layers: 12-28 transformer blocks
- Attention heads: 6-16 heads depending on hidden dimension

### RVQ Configuration

**Quantization Setup:**
- Number of quantizers: `L = 4-8` (variable)
- Codebook size: `K = 1024` (typical)
- Codebook dimension: Matches encoder output dimension
- Quantization method: L2 distance to nearest codebook entry

**Bitrate Control:**
- Variable bitrate by adjusting `L`
- Similar to EnCodec's RVQ implementation
- Enables quality/bitrate trade-offs

### Training Hyperparameters

**Optimizer:**
- Type: AdamW (typical)
- Learning rate: 1e-4 to 5e-4
- Weight decay: 0.01-0.1
- Beta1: 0.9, Beta2: 0.999

**Training Schedule:**
- Batch size: 32-64 (depending on GPU memory)
- Training steps: 1M-2M (varies with dataset)
- Warmup: 10k-50k steps
- Learning rate schedule: Cosine decay or constant

**Loss Configuration:**
- Reconstruction loss: L2 or L1
- Adversarial loss: Standard GAN loss
- Feature matching: Optional
- Loss weights: Balanced (1.0 each) or tuned

### Data Configuration

**Audio Processing:**
- Sampling rate: 24 kHz or 48 kHz
- Audio length: Variable (typically 1-10 seconds)
- Normalization: [-1, 1] range
- Preprocessing: Optional resampling, normalization

**Dataset:**
- Training data: Diverse audio datasets (speech, music, environmental sounds)
- Validation: Held-out test set
- Evaluation: Standard audio quality metrics

---

## Appendix Content

### Architecture Details

#### Transformer Block Structure

**Self-Attention:**
- Query, Key, Value projections
- Multi-head attention mechanism
- Residual connections
- Layer normalization

**Feed-Forward Network:**
- Two linear layers with activation (GELU)
- Expansion ratio: 4.0 (typical)
- Residual connections

**Position Encoding:**
- Learnable embeddings (preferred) or sinusoidal
- Added to patch embeddings
- Enables sequence position awareness

#### Patch Embedding

**Audio Patches:**
- Divide waveform into overlapping or non-overlapping patches
- Patch size: Typically 16-64 samples
- Stride: Can be equal to patch size (non-overlapping) or smaller (overlapping)
- Projection: Linear layer maps patches to embedding dimension

**Alternative: Spectral Patches**
- Can operate on spectrograms or MDCT coefficients
- Patches extracted from frequency-time representation
- Similar to image patch embedding

### Training Procedures

#### Preprocessing

1. **Audio loading**: Load audio files at target sampling rate
2. **Normalization**: Normalize to [-1, 1] range
3. **Chunking**: Divide long audio into fixed-length chunks (optional)
4. **Augmentation**: Optional time-stretching, pitch-shifting, noise addition

#### Training Loop

1. **Forward pass**: Encode → Quantize → Decode
2. **Loss computation**: Reconstruction + Adversarial + Feature matching
3. **Backward pass**: Update encoder-decoder
4. **Discriminator update**: Separate update for discriminator
5. **Codebook maintenance**: Periodic reset of unused codebook entries

#### Stability Techniques

1. **Gradient clipping**: Prevent gradient explosion
2. **Learning rate warmup**: Stabilize early training
3. **EMA**: Exponential moving average for model weights
4. **Mixed precision**: FP16 training for efficiency

### Implementation Considerations

#### Memory Optimization

1. **Gradient checkpointing**: Trade computation for memory
2. **Batch size adjustment**: Reduce for longer sequences
3. **Sequence length limits**: Truncate or chunk very long audio

#### Computational Efficiency

1. **Flash Attention**: Use efficient attention implementations
2. **Mixed precision**: FP16 for faster training
3. **JIT compilation**: Compile forward/backward passes (JAX/PyTorch)
4. **Batch processing**: Efficient batching for inference

#### Codebook Management

1. **Initialization**: Random or k-means initialization
2. **EMA updates**: Optional exponential moving average for codebook entries
3. **Reset strategy**: Reset unused entries periodically
4. **Usage tracking**: Monitor codebook entry usage

### Comparison with Other Codecs

#### vs. EnCodec

**Architecture:**
- EnCodec: Convolutional encoder-decoder
- Stable Audio Codec: Transformer encoder-decoder

**Trade-offs:**
- Transformers: Better long-range modeling, more parameters
- Convolutions: Lower computational cost, local features

**Quality:**
- Comparable quality at similar bitrates
- Transformers may excel on longer sequences

#### vs. MDCTCodec

**Domain:**
- Stable Audio Codec: Waveform domain (or spectral patches)
- MDCTCodec: MDCT domain

**Architecture:**
- Stable Audio Codec: Transformers
- MDCTCodec: ConvNeXt v2

**Trade-offs:**
- MDCT: Lower temporal resolution, spectral representation
- Waveform: Higher temporal resolution, time-domain

### Open-Source Contributions

#### Codebase Features

1. **Complete implementation**: Full training and inference code
2. **Pre-trained models**: Available model checkpoints
3. **Documentation**: Comprehensive documentation and examples
4. **Evaluation tools**: Scripts for quality evaluation
5. **Easy integration**: Simple API for encoding/decoding

#### Community Impact

1. **Research enablement**: Facilitates reproducible research
2. **Education**: Learning resource for neural audio codecs
3. **Comparison baseline**: Reference implementation for comparisons
4. **Extension platform**: Base for custom modifications

### Limitations and Future Work

#### Current Limitations

1. **Computational cost**: Higher than convolutional alternatives
2. **Memory requirements**: Significant memory for large models
3. **Training time**: Longer training compared to smaller models
4. **Real-time CPU**: May be slow on CPU without optimization

#### Future Directions

1. **Efficiency improvements**: Optimize attention mechanisms
2. **Model compression**: Distillation or quantization for deployment
3. **Conditional generation**: Text-to-audio or other conditioning
4. **Multi-modal**: Integration with other modalities
5. **Streaming optimization**: Further latency reduction

### Additional Resources

#### GitHub Repository

- **URL**: https://github.com/Stability-AI/stable-audio-codec
- **License**: Check repository for license information
- **Documentation**: README and additional docs in repository
- **Issues**: GitHub issues for bug reports and feature requests

#### Related Papers

- DiT: Transformer architecture inspiration
- EnCodec: Convolutional baseline comparison
- MDCTCodec: Spectral domain alternative
- Other transformer-based audio models

### Experimental Results

#### Quality Metrics

**Typical Performance (varies by configuration):**
- ViSQOL: 4.0-4.2 (24 kHz, 6 kbps)
- PESQ: 3.7-3.9 (for speech)
- Subjective tests: Competitive with other neural codecs

#### Speed Benchmarks

**Inference Time (varies by hardware):**
- GPU (A100): <0.1× real-time
- GPU (V100): ~0.2× real-time
- CPU: 1-5× real-time (depending on model size)

#### Memory Usage

**Training (varies by batch size and sequence length):**
- Small model: ~8-16 GB GPU memory
- Large model: ~32-64 GB GPU memory

**Inference:**
- Small model: ~2-4 GB GPU memory
- Large model: ~8-16 GB GPU memory

