# MDCTCodec: key equations and contributions

Source: "MDCTCodec: A Lightweight MDCT-Based Neural Audio Codec Towards High Sampling Rate and Low Bitrate Scenarios" (Jiang, Ai, Zheng, Du, Lu, Ling), arXiv: https://arxiv.org/abs/2411.00464

## Notation
- Audio waveform: `x[n]` (time-domain samples)
- MDCT spectrum: `X[k]` (frequency-domain coefficients)
- Window function: `w[n]` (typically sine or Kaiser-Bessel)
- Window size: `N` (typically 2048 or 4096 samples)
- Hop size: `N/2` (50% overlap)
- Encoder output: `z` (continuous latent code)
- Quantized code: `\hat{z}` (discrete latent after RVQ)
- Decoded MDCT: `\hat{X}[k]`
- Reconstructed audio: `\hat{x}[n]`

## MDCT Transform

The Modified Discrete Cosine Transform (MDCT) converts audio to spectral representation:

**Forward MDCT:**
$$
X[k] = \sum_{n=0}^{2N-1} x[n]\,w[n]\,\cos\left(\frac{\pi}{N}\left(n + \frac{N}{2} + \frac{1}{2}\right)\left(k + \frac{1}{2}\right)\right), \quad k = 0, \ldots, N-1. \tag{1}
$$

**Inverse MDCT (IMDCT):**
$$
\hat{x}[n] = \frac{2}{N}\sum_{k=0}^{N-1} \hat{X}[k]\,\cos\left(\frac{\pi}{N}\left(n + \frac{N}{2} + \frac{1}{2}\right)\left(k + \frac{1}{2}\right)\right), \quad n = 0, \ldots, 2N-1. \tag{2}
$$

The window function `w[n]` satisfies the perfect reconstruction condition:
$$
w^2[n] + w^2[n+N] = 1.
$$

## Architecture Overview

MDCTCodec operates entirely in the MDCT domain, avoiding direct waveform manipulation:

1. **Encoding**: `X[k] \rightarrow z \rightarrow \hat{z}` (MDCT → continuous latent → quantized)
2. **Decoding**: `\hat{z} \rightarrow \hat{X}[k] \rightarrow \hat{x}[n]` (quantized → MDCT → audio)

This reduces temporal resolution significantly compared to waveform-based codecs, enabling efficient compression.

## Encoder

The encoder `E_\theta` maps MDCT spectrum to continuous latent code:

$$
z = E_\theta(X[k]). \tag{3}
$$

**Architecture**: Modified ConvNeXt v2 backbone
- Input: MDCT spectrum `X[k]` of shape `(T, N)` where `T` is number of frames
- Output: Continuous latent `z` of shape `(T', D)` with reduced temporal dimension `T' < T`

The ConvNeXt v2 modifications enable efficient processing of spectral features while maintaining translation equivariance.

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
\hat{z} = \sum_{i=1}^{L} \hat{z}_i. \tag{4}
$$

Each quantizer `\mathrm{Quantize}_i` uses a codebook `C_i` of size `K`:
$$
\mathrm{Quantize}_i(z) = \arg\min_{c \in C_i} \|z - c\|^2.
$$

The final quantized representation is the sum of all residual quantizations. This hierarchical quantization enables fine-grained control over bitrate by varying `L`.

## Decoder

The decoder `D_\phi` reconstructs MDCT spectrum from quantized codes:

$$
\hat{X}[k] = D_\phi(\hat{z}). \tag{5}
$$

**Architecture**: Modified ConvNeXt v2 backbone (symmetric to encoder)
- Input: Quantized latent `\hat{z}` of shape `(T', D)`
- Output: Reconstructed MDCT spectrum `\hat{X}[k]` of shape `(T, N)`

The decoder upsamples temporally to match the original MDCT frame rate.

## Multi-Resolution MDCT-based Discriminator (MR-MDCTD)

The discriminator `D_\psi` operates on MDCT spectra at multiple resolutions to provide adversarial training signal:

**Multi-resolution discrimination:**
$$
D_\psi^{(r)}(\hat{X}[k]) = \text{ConvNeXt}(\text{Downsample}_r(\hat{X}[k])), \quad r \in \{1, 2, 4, 8\}, \tag{6}
$$

where `\text{Downsample}_r` reduces temporal resolution by factor `r`.

**Discriminator loss:**
$$
\mathcal{L}_D = \mathbb{E}_{X \sim p_{\text{data}}}[\log D_\psi(X)] + \mathbb{E}_{\hat{X} \sim p_\theta}[\log(1 - D_\psi(\hat{X}))]. \tag{7}
$$

Operating in MDCT domain (rather than waveform) allows the discriminator to focus on spectral quality, which is more aligned with perceptual audio quality.

## Training Objectives

**Reconstruction loss:**
$$
\mathcal{L}_{\text{recon}} = \mathbb{E}_{X \sim p_{\text{data}}} \|X - \hat{X}\|^2. \tag{8}
$$

**Adversarial loss (generator):**
$$
\mathcal{L}_{\text{adv}} = -\mathbb{E}_{\hat{X} \sim p_\theta}[\log D_\psi(\hat{X})]. \tag{9}
$$

**Feature matching loss:**
$$
\mathcal{L}_{\text{fm}} = \mathbb{E}_{X \sim p_{\text{data}}} \sum_{l} \|D_\psi^{(l)}(X) - D_\psi^{(l)}(\hat{X})\|^2, \tag{10}
$$

where `D_\psi^{(l)}` denotes intermediate features from discriminator layer `l`.

**Total generator loss:**
$$
\mathcal{L}_G = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{fm}}\mathcal{L}_{\text{fm}}. \tag{11}
$$

## Key Contributions

1. **MDCT-domain processing**: Operates entirely on MDCT spectrum, avoiding high temporal resolution of waveforms. This enables:
   - Lower computational complexity
   - Fewer downsampling/upsampling operations
   - Reduced model parameters

2. **Lightweight architecture**: Modified ConvNeXt v2 backbone provides efficient feature extraction with fewer parameters than transformer-based alternatives.

3. **Multi-resolution discriminator**: MR-MDCTD operates at multiple temporal resolutions in MDCT domain, providing better adversarial training signal than single-resolution discriminators.

4. **High sampling rate support**: Achieves high quality at 48 kHz sampling rate with low bitrate (6 kbps), outperforming waveform-based codecs.

5. **Efficient training and inference**: Fast training and generation (123× real-time on GPU, 16.9× on CPU) due to reduced temporal resolution and efficient architecture.

## Performance

- **Sampling rate**: 48 kHz
- **Bitrate**: 6 kbps
- **Quality metric**: ViSQOL score of 4.18 on VCTK corpus
- **Speed**: 123× real-time on GPU, 16.9× on CPU
- **Model size**: Lightweight (fewer parameters than baseline codecs)

## Advantages over Baseline Codecs

1. **vs. Waveform-based codecs** (SoundStream, Encodec, HiFi-Codec):
   - Better handling of high sampling rates (48 kHz)
   - Lower computational cost
   - Fewer model parameters

2. **vs. Other spectral codecs** (APCodec, MDCTNet):
   - Lower bitrate (6 kbps vs. 20-32 kbps for MDCTNet)
   - Simpler architecture (no recurrent structures)
   - Better training efficiency

3. **vs. DAC**:
   - Lower bitrate requirement (6 kbps vs. 8 kbps)
   - Faster generation speed

## Implementation Details for Reimplementation

### ConvNeXt v2 Backbone Modifications

The encoder and decoder use a modified ConvNeXt v2 architecture:
- **Activation**: GELU (Gaussian Error Linear Units) instead of ReLU
- **Normalization**: LayerNorm instead of BatchNorm (better for variable-length sequences)
- **Downsampling**: Strided convolutions for temporal reduction in encoder
- **Upsampling**: Transposed convolutions or interpolation for temporal expansion in decoder
- **Depth**: Multiple stages with increasing channel dimensions

### RVQ Implementation Details

- **Codebook size**: `K = 1024` (typical)
- **Number of quantizers**: `L` varies with target bitrate (e.g., `L = 4` for 6 kbps)
- **Codebook initialization**: Random initialization or k-means on encoder outputs
- **Quantization**: Uses straight-through estimator (STE) for gradient flow:
  $$
  \frac{\partial \hat{z}_i}{\partial z_{i-1}} \approx 1 \text{ (during backprop)}
  $$

### Training Procedure

1. **Preprocessing**:
   - Audio resampled to 48 kHz
   - MDCT computed with window size `N = 2048` or `4096`
   - 50% overlap (hop size `N/2`)

2. **Training schedule**:
   - Alternating updates: discriminator and generator updated separately
   - Learning rate: Typically `1e-4` for generator, `4e-4` for discriminator
   - Optimizer: AdamW with weight decay
   - Batch size: Varies (typically 16-32)

3. **Loss weighting**:
   - `λ_recon = 1.0` (reconstruction)
   - `λ_adv = 1.0` (adversarial)
   - `λ_fm = 1.0` (feature matching)

### Multi-Resolution Discriminator Architecture

The MR-MDCTD processes MDCT spectra at resolutions `r ∈ {1, 2, 4, 8}`:
- Each resolution uses a separate ConvNeXt v2 branch
- Downsampling via average pooling or strided convolution
- Final discriminator output: average of all resolution outputs
- This provides multi-scale adversarial feedback

### Bitrate Calculation

For a given configuration:
$$
\text{Bitrate} = \frac{L \times \log_2(K) \times T' \times f_s}{T \times N/2} \text{ bps}
$$

where:
- `L`: number of RVQ quantizers
- `K`: codebook size
- `T'`: number of latent frames (after encoder downsampling)
- `T`: number of MDCT frames
- `f_s`: sampling rate
- `N/2`: hop size (frames per second = `f_s / (N/2)`)

### Key Implementation Considerations

1. **MDCT/IMDCT**: Must ensure perfect reconstruction property
   - Window function must satisfy Princen-Bradley condition
   - Proper overlap-add in IMDCT reconstruction

2. **Gradient flow**: RVQ uses straight-through estimator to allow gradients through quantization

3. **Memory efficiency**: MDCT domain reduces temporal resolution, enabling larger batch sizes

4. **JAX-specific considerations**:
   - Use `jax.lax.scan` for sequential RVQ quantization (if needed)
   - Vectorize MDCT/IMDCT operations across frames
   - Use `vmap` for batch processing
   - Consider `jax.jit` compilation for encoder/decoder

### Architecture Dimensions (Typical)

For 48 kHz audio with `N = 2048`:
- **Input MDCT**: `(T, 1024)` where `T ≈ (duration × 48000) / 1024`
- **Encoder output**: `(T', D)` where `T' ≈ T/4` to `T/8`, `D = 128` to `512`
- **Quantized latent**: Same shape as encoder output
- **Decoder output**: `(T, 1024)` (reconstructed MDCT)
- **Final audio**: `(duration × 48000,)` samples

### Training Tricks

1. **Spectral loss**: Can add frequency-weighted reconstruction loss to emphasize perceptually important bands
2. **Commitment loss**: Add `β ||z - \hat{z}||^2` to encourage encoder to produce quantizable outputs
3. **Codebook reset**: Periodically reset unused codebook entries to nearest encoder outputs
4. **Progressive training**: Start with fewer quantizers and gradually increase `L`

