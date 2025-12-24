# MDCT Theory

The **MDCT (Modified Discrete Cosine Transform)** is a key component of this audio encoder. It transforms time-domain audio signals into frequency-domain spectral representations, enabling efficient compression and encoding.

## Why MDCT?

MDCT is widely used in audio codecs (e.g., MP3, AAC) because it:

- Provides **perfect reconstruction** when combined with proper windowing
- Enables **50% overlap** between frames for smooth reconstruction
- Reduces **temporal resolution** compared to waveform-based approaches, making compression more efficient
- Operates entirely in the **spectral domain**, which is more suitable for neural audio codecs

## Implementation

Our MDCT implementation (`meanflow_audio_codec/preprocessing/mdct.py`) includes:

- **Forward MDCT**: Converts audio waveforms to MDCT coefficients
- **Inverse MDCT (IMDCT)**: Reconstructs audio from MDCT coefficients
- **FFT-based optimization**: O(N log N) complexity for large window sizes (≥512)
- **Direct cosine fallback**: O(N²) implementation for smaller windows or Metal backend
- **Sine window**: Default window function satisfying the Princen-Bradley perfect reconstruction condition

## Usage Example

```python
from meanflow_audio_codec.preprocessing.mdct import mdct, imdct

# Transform audio to MDCT domain
audio_samples = ...  # Shape: (batch, time_samples)
mdct_coeffs = mdct(audio_samples, window_size=2048)  # Shape: (batch, num_frames, window_size)

# Reconstruct audio from MDCT coefficients
reconstructed_audio = imdct(mdct_coeffs, window_size=2048)
```

## Audio Encoder Pipeline

The audio encoder operates as follows:

1. **Audio → MDCT**: Transform input audio to MDCT spectral domain
2. **MDCT → Latent**: Encode MDCT coefficients using Improved Mean Flow
3. **Latent → MDCT**: Decode latent codes back to MDCT coefficients
4. **MDCT → Audio**: Reconstruct audio using inverse MDCT

This approach follows the MDCTCodec architecture (see [MDCTCodec Key Equations](mdctcodec_key_eqn.md)), where the entire encoding/decoding process operates in the MDCT domain rather than directly on waveforms.

## Perfect Reconstruction

The MDCT achieves perfect reconstruction when:

1. The window function satisfies the Princen-Bradley condition:
   $$w^2[n] + w^2[n+N] = 1$$
   where $N$ is the window size.

2. The hop size is exactly half the window size (50% overlap).

3. Proper overlap-add is performed during inverse MDCT reconstruction.

## See Also

- [MDCTCodec Key Equations](mdctcodec_key_eqn.md) - MDCTCodec architecture details
- [MDCT Optimization](../implementation/performance/mdct_optimization.md) - Performance analysis
- [API Reference](../../developer_guide/api_reference.md) - MDCT API documentation

