# Research Documentation

This section contains the mathematical foundations and theoretical background for the Meanflow Audio Codec project.

## Contents

### Core Methods

#### MeanFlow
- [MeanFlow Key Equations](meanflow/meanflow_key_eqn.md) - Mathematical formulation of MeanFlow

#### Improved MeanFlow
- [Improved MeanFlow Key Equations](improved_meanflow/improved_meanflow_key_eqn.md) - Mathematical formulation of Improved MeanFlow (iMF)

### Architecture References

#### DiT
- [DiT Key Equations](dit/dit_key_eqn.md) - Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)

#### SiT
- [SiT Key Equations](sit/sit_key_eqn.md) - Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (Ma et al., 2024)

### Audio Codecs

#### MDCT
- [MDCT Theory](mdct/theory.md) - MDCT fundamentals and implementation details
- [MDCTCodec Key Equations](mdct/mdctcodec_key_eqn.md) - MDCTCodec architecture and equations

#### EnCodec
- [EnCodec Key Equations](encodec/encodec_key_eqn.md) - High Fidelity Neural Audio Compression (Défossez et al., 2022)

#### Stable Audio Codec
- [Stable Audio Codec Key Equations](stable_audio_codec/stable_audio_codec_key_eqn.md) - Stability AI Open Audio Codec (2024)

### Fundamentals

#### Digital Audio Principles
- [Digital Audio Key Concepts](digital_audio/digital_audio_key_eqn.md) - Principles of Digital Audio (Pohlmann, 2010)

## Overview

This project implements:
- **Improved Mean Flow (iMF)**: A method for fast one-step generative modeling
- **MDCT-based encoding**: Audio encoding in the Modified Discrete Cosine Transform domain
- **Flow Matching**: Training objective for generative models

## References

### Core Methods
- MeanFlow: "Mean Flows for One-step Generative Modeling" (Geng et al., 2025)
- Improved MeanFlow: "Improved Mean Flows: On the Challenges of Fastforward Generative Models" (Geng et al., 2024)

### Architecture References
- DiT: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022)
- SiT: "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers" (Ma et al., 2024)

### Audio Codec References
- MDCTCodec: "MDCTCodec: A Lightweight MDCT-Based Neural Audio Codec Towards High Sampling Rate and Low Bitrate Scenarios" (Jiang et al., 2024)
- EnCodec: "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
- Stable Audio Codec: Stability AI Open Audio Codec (2024)

### Fundamentals
- Principles of Digital Audio: "Principles of Digital Audio" (Pohlmann, 2010, 6th edition)

