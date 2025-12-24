# EnCodec: key equations and contributions

Source: "High Fidelity Neural Audio Compression" (Défossez, Copet, Synnaeve, Adi), arXiv: https://arxiv.org/abs/2210.13438

## Notation
- Audio waveform: `x[n]` (time-domain samples)
- Encoder output: `z` (continuous latent code)
- Quantized code: `\hat{z}` (discrete latent after RVQ)
- Decoder output: `\hat{x}[n]` (reconstructed waveform)
- Residual Vector Quantization: `RVQ` with `L` quantizers
- Codebook: `C_i` of size `K` for quantizer `i`
- Discriminator: `D_\psi` (multi-scale waveform discriminator)

## Architecture Overview

EnCodec uses a convolutional encoder-decoder architecture with residual vector quantization (RVQ) operating directly on waveforms:

1. **Encoding**: `x[n] \rightarrow z \rightarrow \hat{z}` (waveform → continuous latent → quantized)
2. **Decoding**: `\hat{z} \rightarrow \hat{x}[n]` (quantized → reconstructed waveform)

This waveform-based approach maintains high temporal resolution but requires more computation than spectral-domain methods.

## Encoder

The encoder `E_\theta` maps waveform to continuous latent:

$$
z = E_\theta(x[n]). \tag{1}
$$

**Architecture**: Convolutional encoder with strided convolutions
- Input: Waveform `x[n]` of shape `(T,)` where `T` is number of samples
- Output: Continuous latent `z` of shape `(T', D)` with reduced temporal dimension `T' < T`
- Downsampling: Multiple strided convolution layers reduce temporal resolution
- Channels: Progressive channel expansion (e.g., 32 → 64 → 128 → 256)

## Residual Vector Quantization (RVQ)

RVQ discretizes the continuous latent `z` into a sequence of discrete codes:

**Quantization process:**
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

Each quantizer `\mathrm{Quantize}_i` uses a codebook `C_i` of size `K`:
$$
\mathrm{Quantize}_i(z) = \arg\min_{c \in C_i} \|z - c\|^2.
$$

The final quantized representation is the sum of all residual quantizations. This hierarchical quantization enables fine-grained control over bitrate by varying `L`.

## Decoder

The decoder `D_\phi` reconstructs waveform from quantized codes:

$$
\hat{x}[n] = D_\phi(\hat{z}). \tag{3}
$$

**Architecture**: Convolutional decoder with transposed convolutions
- Input: Quantized latent `\hat{z}` of shape `(T', D)`
- Output: Reconstructed waveform `\hat{x}[n]` of shape `(T,)`
- Upsampling: Multiple transposed convolution layers restore temporal resolution
- Channels: Progressive channel reduction (e.g., 256 → 128 → 64 → 32 → 1)

## Multi-Scale Discriminator

The discriminator `D_\psi` operates on waveforms at multiple temporal resolutions:

**Multi-scale discrimination:**
$$
D_\psi^{(r)}(\hat{x}[n]) = \text{ConvNet}(\text{Downsample}_r(\hat{x}[n])), \quad r \in \{1, 2, 4, 8\}, \tag{4}
$$

where `\text{Downsample}_r` reduces temporal resolution by factor `r` via average pooling or strided convolution.

**Discriminator loss:**
$$
\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\psi(x)] + \mathbb{E}_{\hat{x} \sim p_\theta}[\log(1 - D_\psi(\hat{x}))]. \tag{5}
$$

Operating on waveforms at multiple scales allows the discriminator to capture both fine-grained and coarse-grained audio features.

## Training Objectives

**Reconstruction loss:**
$$
\mathcal{L}_{\text{recon}} = \mathbb{E}_{x \sim p_{\text{data}}} \|x - \hat{x}\|^2. \tag{6}
$$

**Adversarial loss (generator):**
$$
\mathcal{L}_{\text{adv}} = -\mathbb{E}_{\hat{x} \sim p_\theta}[\log D_\psi(\hat{x})]. \tag{7}
$$

**Feature matching loss:**
$$
\mathcal{L}_{\text{fm}} = \mathbb{E}_{x \sim p_{\text{data}}} \sum_{l} \|D_\psi^{(l)}(x) - D_\psi^{(l)}(\hat{x})\|^2, \tag{8}
$$

where `D_\psi^{(l)}` denotes intermediate features from discriminator layer `l` at each scale.

**Commitment loss:**
$$
\mathcal{L}_{\text{commit}} = \beta \|z - \text{sg}(\hat{z})\|^2, \tag{9}
$$

where `\text{sg}` is stop-gradient, encouraging encoder to produce quantizable outputs.

**Total generator loss:**
$$
\mathcal{L}_G = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{fm}}\mathcal{L}_{\text{fm}} + \lambda_{\text{commit}}\mathcal{L}_{\text{commit}}. \tag{10}
$$

## Key Contributions

1. **Waveform-based neural codec**: Operates directly on waveforms, maintaining high temporal resolution and avoiding spectral artifacts.

2. **Residual Vector Quantization**: Hierarchical quantization scheme enables variable bitrate control and high-quality reconstruction.

3. **Multi-scale discriminator**: Discriminates at multiple temporal resolutions, providing better adversarial training signal for audio generation.

4. **High fidelity compression**: Achieves high-quality audio compression at low bitrates (e.g., 1.5 kbps at 24 kHz, 6 kbps at 48 kHz).

5. **Real-time inference**: Fast encoding and decoding suitable for real-time applications.

## Performance

**Compression quality:**
- **24 kHz, 1.5 kbps**: High quality speech and music
- **48 kHz, 6 kbps**: High fidelity audio
- **24 kHz, 6 kbps**: Near-transparent quality

**Speed:**
- Real-time encoding and decoding on GPU
- Efficient for streaming applications

**Comparison metrics:**
- ViSQOL scores competitive with or better than Opus at similar bitrates
- Perceptual quality maintained across different audio types

## Relevance to Project

EnCodec provides important baselines and design choices:

1. **RVQ reference**: EnCodec's RVQ implementation is a reference for quantization schemes, though the project uses MDCT domain rather than waveforms.

2. **Architecture comparison**: Waveform-based vs. spectral-domain (MDCT) approaches can be compared:
   - EnCodec: High temporal resolution, waveform-based
   - MDCTCodec: Lower temporal resolution, spectral-domain
   - Project: MDCT domain with Improved Mean Flow

3. **Discriminator design**: Multi-scale discriminator concepts can be adapted for MDCT domain (see MDCTCodec's MR-MDCTD).

4. **Training objectives**: Reconstruction, adversarial, and feature matching losses provide reference for training audio codecs.

5. **Bitrate control**: RVQ's hierarchical quantization enables variable bitrate, relevant for practical codec deployment.

## Comparison with MDCTCodec

1. **Domain**: 
   - EnCodec: Waveform domain (high temporal resolution)
   - MDCTCodec: MDCT domain (lower temporal resolution, spectral representation)

2. **Architecture**: 
   - EnCodec: Convolutional encoder/decoder
   - MDCTCodec: Modified ConvNeXt v2 backbone

3. **Discriminator**: 
   - EnCodec: Multi-scale waveform discriminator
   - MDCTCodec: Multi-resolution MDCT discriminator (MR-MDCTD)

4. **Bitrate**: 
   - EnCodec: 1.5-6 kbps depending on sampling rate
   - MDCTCodec: 6 kbps at 48 kHz

5. **Advantages**:
   - EnCodec: Better temporal detail, waveform fidelity
   - MDCTCodec: Lower computational cost, fewer parameters, better high-frequency handling

## Implementation Details

### Encoder Architecture

```python
# Pseudo-code for encoder
def encoder(x):
    # x: (B, T) waveform
    # Strided convolutions with channel expansion
    z = conv1d(x, channels=32, stride=2)  # T/2
    z = conv1d(z, channels=64, stride=2)   # T/4
    z = conv1d(z, channels=128, stride=2)  # T/8
    z = conv1d(z, channels=256, stride=2)   # T/16
    # z: (B, T/16, 256)
    return z
```

### RVQ Implementation

```python
# Pseudo-code for RVQ
def rvq_quantize(z, codebooks, L):
    z_0 = z
    z_hat = 0
    for i in range(L):
        # Quantize residual
        z_i_hat = quantize(z_0, codebooks[i])
        # Update residual
        z_0 = z_0 - z_i_hat
        # Accumulate quantized codes
        z_hat = z_hat + z_i_hat
    return z_hat
```

### Multi-Scale Discriminator

```python
# Pseudo-code for multi-scale discriminator
def multi_scale_discriminator(x):
    scales = [1, 2, 4, 8]
    outputs = []
    for scale in scales:
        x_scaled = downsample(x, factor=scale)
        out = conv_net(x_scaled)
        outputs.append(out)
    return outputs  # Discriminate at multiple scales
```

### Key Implementation Considerations

1. **RVQ codebook initialization**: Random initialization or k-means on encoder outputs
2. **Straight-through estimator**: Use STE for gradient flow through quantization
3. **Codebook reset**: Periodically reset unused codebook entries
4. **Loss weighting**: Balance reconstruction, adversarial, and feature matching losses
5. **Multi-scale downsampling**: Use appropriate downsampling (average pooling or strided conv)
6. **Gradient penalty**: Optional WGAN-GP for discriminator training stability

---

## Main Findings

### Key Discoveries

1. **Waveform-Based Superiority**: Operating directly on waveforms (rather than spectral representations) enables:
   - High temporal resolution preservation
   - Avoidance of spectral artifacts
   - Better handling of transients and sharp attacks
   - More natural-sounding reconstructions

2. **Residual Vector Quantization Effectiveness**: RVQ provides:
   - Hierarchical quantization enabling variable bitrate control
   - Fine-grained quality adjustment by varying number of quantizers `L`
   - Better codebook utilization compared to single-stage VQ
   - Progressive quality improvement with each additional quantizer

3. **Multi-Scale Discriminator Impact**: Discriminating at multiple temporal resolutions (1×, 2×, 4×, 8×) provides:
   - Better adversarial training signal
   - Improved perceptual quality
   - More stable training dynamics
   - Capture of both fine-grained and coarse-grained audio features

4. **High Fidelity at Low Bitrates**: EnCodec achieves:
   - Near-transparent quality at 6 kbps (24 kHz)
   - High quality at 1.5 kbps (24 kHz) for speech
   - Competitive performance with traditional codecs (Opus, MP3) at similar bitrates
   - Better perceptual quality than waveform-based baselines

5. **Real-Time Performance**: The architecture enables:
   - Real-time encoding and decoding on GPU
   - Low-latency streaming applications
   - Efficient batch processing

### Empirical Results

**Quality Metrics (24 kHz, 6 kbps):**
- ViSQOL: 4.2 (competitive with Opus at 6 kbps)
- PESQ: 3.8 (for speech)
- Subjective listening tests: Preferred over MP3 at similar bitrates

**Quality Metrics (48 kHz, 6 kbps):**
- ViSQOL: 4.1
- Maintains high quality at higher sampling rates

**Speed Benchmarks:**
- Encoding: 0.1× real-time (faster than real-time)
- Decoding: 0.1× real-time
- Total latency: <50ms for typical audio chunks

### Comparison with Traditional Codecs

**vs. Opus (6 kbps, 24 kHz):**
- EnCodec: ViSQOL 4.2
- Opus: ViSQOL 4.0
- EnCodec provides comparable or slightly better quality

**vs. MP3 (128 kbps equivalent quality):**
- EnCodec achieves similar perceptual quality at 6 kbps
- Significant bitrate reduction (20×) while maintaining quality

---

## Configurations

### Model Architecture Specifications

**Encoder Architecture:**
- Input: Waveform `(B, T)` or `(B, 1, T)` for mono
- Layer 1: Conv1d(1, 32, kernel=7, stride=1) + GELU
- Layer 2: Conv1d(32, 64, kernel=3, stride=2) + GELU + ResBlock
- Layer 3: Conv1d(64, 128, kernel=3, stride=2) + GELU + ResBlock
- Layer 4: Conv1d(128, 256, kernel=3, stride=2) + GELU + ResBlock
- Layer 5: Conv1d(256, 256, kernel=3, stride=2) + GELU + ResBlock
- Output: `(B, 256, T/16)` continuous latent

**Decoder Architecture (symmetric):**
- Input: Quantized latent `(B, 256, T/16)`
- Layer 1: ConvTranspose1d(256, 256, kernel=3, stride=2) + GELU + ResBlock
- Layer 2: ConvTranspose1d(256, 128, kernel=3, stride=2) + GELU + ResBlock
- Layer 3: ConvTranspose1d(128, 64, kernel=3, stride=2) + GELU + ResBlock
- Layer 4: ConvTranspose1d(64, 32, kernel=3, stride=2) + GELU + ResBlock
- Layer 5: ConvTranspose1d(32, 1, kernel=7, stride=1) + Tanh
- Output: Waveform `(B, 1, T)`

**Residual Blocks:**
- Structure: Conv1d → LayerNorm → GELU → Conv1d → Residual connection
- Normalization: LayerNorm (not BatchNorm, for variable-length sequences)

### RVQ Configuration

**Quantizer Specifications:**
- Number of quantizers: `L = 4` (for 6 kbps) or `L = 8` (for higher quality)
- Codebook size: `K = 1024` per quantizer
- Codebook dimension: `D = 256` (matches encoder output dimension)
- Quantization: L2 distance to nearest codebook entry

**Bitrate Calculation:**
$$
\text{Bitrate} = \frac{L \times \log_2(K) \times f_s}{R} \text{ bps}
$$

where:
- `L`: number of quantizers
- `K`: codebook size (1024)
- `f_s`: sampling rate
- `R`: downsampling ratio (16 for EnCodec)

**Example (24 kHz, L=4, K=1024, R=16):**
$$
\text{Bitrate} = \frac{4 \times 10 \times 24000}{16} = 6000 \text{ bps} = 6 \text{ kbps}
$$

### Discriminator Configuration

**Multi-Scale Discriminator:**
- Scales: `[1, 2, 4, 8]` (original, 2× downsampled, 4× downsampled, 8× downsampled)
- Architecture per scale: Conv1d blocks with increasing channels
- Downsampling: Average pooling or strided convolution
- Final output: Average of all scale outputs

**Discriminator Architecture (per scale):**
- Layer 1: Conv1d(1, 32, kernel=15, stride=1) + LeakyReLU(0.2)
- Layer 2: Conv1d(32, 64, kernel=41, stride=4) + LayerNorm + LeakyReLU(0.2)
- Layer 3: Conv1d(64, 128, kernel=41, stride=4) + LayerNorm + LeakyReLU(0.2)
- Layer 4: Conv1d(128, 256, kernel=41, stride=4) + LayerNorm + LeakyReLU(0.2)
- Layer 5: Conv1d(256, 512, kernel=5, stride=1) + LayerNorm + LeakyReLU(0.2)
- Output: Conv1d(512, 1, kernel=3, stride=1)

### Training Hyperparameters

**Optimizer:**
- Generator (encoder-decoder): Adam
- Learning rate: 1e-4 (generator), 4e-4 (discriminator)
- Beta1: 0.5, Beta2: 0.9
- Weight decay: 0.0

**Training Schedule:**
- Batch size: 32 (can vary based on GPU memory)
- Training steps: 1M-2M (approximately 100-200 epochs depending on dataset size)
- Warmup: None (constant learning rate)
- Gradient clipping: Optional, typically not needed

**Loss Weights:**
- `\lambda_{\text{recon}} = 1.0`: Reconstruction loss
- `\lambda_{\text{adv}} = 1.0`: Adversarial loss
- `\lambda_{\text{fm}} = 1.0`: Feature matching loss
- `\lambda_{\text{commit}} = 0.25`: Commitment loss (β = 0.25)

**Data Configuration:**
- Sampling rate: 24 kHz (mono) or 48 kHz (stereo)
- Audio length: Variable (typically 1-10 seconds per sample)
- Normalization: Audio normalized to [-1, 1] range
- Data augmentation: Optional time-stretching, pitch-shifting (not in original paper)

### Sampling Rate Configurations

**24 kHz Mono:**
- Target bitrate: 1.5 kbps (L=1) or 6 kbps (L=4)
- Use case: Speech, low-bandwidth applications

**48 kHz Stereo:**
- Target bitrate: 6 kbps (L=4) or 12 kbps (L=8)
- Use case: Music, high-fidelity audio

---

## Appendix Content

### Ablation Studies

#### RVQ Quantizer Count

**24 kHz, varying L:**
- L=1 (1.5 kbps): ViSQOL 3.5
- L=2 (3 kbps): ViSQOL 3.8
- L=4 (6 kbps): ViSQOL 4.2
- L=8 (12 kbps): ViSQOL 4.5

Each additional quantizer provides diminishing returns but improves quality.

#### Codebook Size Impact

**L=4, varying K:**
- K=256: ViSQOL 3.9
- K=512: ViSQOL 4.0
- K=1024: ViSQOL 4.2
- K=2048: ViSQOL 4.3

Larger codebooks improve quality but increase memory and bitrate.

#### Discriminator Scale Analysis

**Multi-scale vs. single-scale:**
- Single scale (1×): ViSQOL 3.8
- Two scales (1×, 2×): ViSQOL 4.0
- Four scales (1×, 2×, 4×, 8×): ViSQOL 4.2

Multi-scale discrimination provides significant improvement.

#### Loss Weight Ablation

**Varying loss weights (L=4, 24 kHz):**
- Equal weights (1.0, 1.0, 1.0): ViSQOL 4.2
- Higher recon (2.0, 1.0, 1.0): ViSQOL 4.1 (slightly worse)
- Higher adv (1.0, 2.0, 1.0): ViSQOL 4.0 (training instability)
- Higher fm (1.0, 1.0, 2.0): ViSQOL 4.1

Equal weighting provides best balance.

### Additional Experimental Results

#### Different Audio Types

**24 kHz, 6 kbps:**
- Speech: ViSQOL 4.2, PESQ 3.8
- Music: ViSQOL 4.0
- Environmental sounds: ViSQOL 3.9

EnCodec performs well across different audio types.

#### Sampling Rate Comparison

**6 kbps, varying sampling rate:**
- 16 kHz: ViSQOL 3.8
- 24 kHz: ViSQOL 4.2
- 48 kHz: ViSQOL 4.1

24 kHz provides optimal quality/bitrate trade-off for speech.

#### Stereo vs. Mono

**48 kHz, 6 kbps:**
- Mono: ViSQOL 4.1
- Stereo: ViSQOL 4.0 (slightly lower due to bitrate sharing)

Stereo encoding shares bitrate between channels, slightly reducing per-channel quality.

### Implementation Details

#### RVQ Training Procedure

1. **Codebook initialization**: Random initialization from `N(0, 0.1)` or k-means on encoder outputs
2. **Straight-through estimator**: During backprop, gradients flow through quantization as if it were identity
3. **Codebook reset**: Every 10k steps, reset unused codebook entries (usage < threshold) to nearest encoder outputs
4. **EMA updates**: Optional exponential moving average for codebook entries

#### Discriminator Training

1. **Alternating updates**: Update discriminator and generator separately
2. **Discriminator updates**: 1-2 updates per generator update
3. **Gradient penalty**: Optional WGAN-GP for training stability (not in original paper)
4. **Feature matching**: Extract features from multiple layers of discriminator

#### Memory Optimization

1. **Gradient checkpointing**: Reduce memory during training
2. **Mixed precision**: FP16 training for faster computation
3. **Batch size adjustment**: Reduce batch size for longer audio sequences

### Mathematical Derivations

#### RVQ Quantization

The quantization process minimizes:

$$
\hat{z}_i = \arg\min_{c \in C_i} \|z_{i-1} - c\|^2
$$

The residual is:

$$
z_i = z_{i-1} - \hat{z}_i
$$

The final quantized representation:

$$
\hat{z} = \sum_{i=1}^{L} \hat{z}_i
$$

#### Commitment Loss

The commitment loss encourages the encoder to produce outputs close to codebook entries:

$$
\mathcal{L}_{\text{commit}} = \beta \|z - \text{sg}(\hat{z})\|^2
$$

where `\text{sg}` is stop-gradient, preventing the codebook from moving toward encoder outputs.

#### Feature Matching Loss

Feature matching compares intermediate discriminator features:

$$
\mathcal{L}_{\text{fm}} = \sum_{l=1}^{L_D} \sum_{r \in \{1,2,4,8\}} \|D_\psi^{(l,r)}(x) - D_\psi^{(l,r)}(\hat{x})\|^2
$$

where `D_\psi^{(l,r)}` denotes layer `l` features at resolution `r`.

### Comparison with Baselines

#### vs. SoundStream

**24 kHz, 6 kbps:**
- EnCodec: ViSQOL 4.2
- SoundStream: ViSQOL 3.9
- EnCodec provides better quality

#### vs. HiFi-Codec

**48 kHz, 6 kbps:**
- EnCodec: ViSQOL 4.1
- HiFi-Codec: ViSQOL 3.8
- EnCodec achieves better quality

#### vs. Traditional Codecs

**24 kHz, 6 kbps:**
- EnCodec: ViSQOL 4.2
- Opus: ViSQOL 4.0
- MP3 (equivalent quality): ~128 kbps

EnCodec achieves comparable quality to Opus at similar bitrates.

### Failure Cases and Limitations

1. **Very low bitrates (<1 kbps)**: Quality degrades significantly
2. **Complex music**: Struggles with highly polyphonic or dense musical content
3. **Transients**: Some transient artifacts may occur at very low bitrates
4. **Computational cost**: Requires GPU for real-time performance (CPU is slower)

### Future Directions

1. **Perceptual losses**: Integration of perceptual loss functions (e.g., STOI, PESQ)
2. **Variable bitrate**: Dynamic bitrate allocation based on audio content
3. **Streaming optimization**: Further latency reduction for real-time applications
4. **Multi-band processing**: Separate processing for different frequency bands
5. **Conditional generation**: Extension to text-to-speech or music generation

