# Meanflow Audio Codec

Most state-of-the-art neural audio codecs start from a rate–distortion autoencoder, then add GAN discriminators to recover perceptual detail at low bitrate. This works, but it trades reconstruction stability for adversarial tuning and failure modes (training instability, sensitivity to discriminator design, and brittle loss balancing).

This repository explores a non-adversarial alternative: treat the decoder as a conditional sampler trained by flow matching, so "perceptual sharpness" comes from learning a distribution $p(x|y)$ rather than forcing a deterministic point estimate $\hat{x}(y)$. Improved Mean Flows (iMF) makes this practical for codec-style deployment by targeting high-quality synthesis with very low NFE, effectively acting as a stochastic post-filter/transport conditioned on the compressed representation. Distilling to a few function evaluations is therefore not "full circle back to an autoencoder": even in the low-NFE limit, the model is trained to reproduce the teacher sampler's distributional behavior across noise levels, and it retains an explicit noise-to-audio transport pathway that a standard AE objective does not enforce.

Concretely, this repo implements an MDCT-domain codec in JAX/Flax and benchmarks the progression (A) pure autoencoder → (B) flow matching → (C) mean flow → (D) improved mean flow, to quantify when iMF can replace adversarial perceptual regularization with a stable regression-style training signal.

## Research Plan

This project follows a systematic experimental approach to establish baseline performances and understand the impact of different design choices. The research plan is structured as follows:

### Method Progression (for each architecture)

For each architecture, we study baseline performances in the following order:

1. **(A) Pure Autoencoder** - Baseline reconstruction performance
2. **(B) Flow Matcher** - Standard flow matching approach
3. **(C) Mean Flow** - Original Mean Flow method
4. **(D) Improved Mean Flow** - Enhanced Mean Flow with improved training dynamics

### Architecture Progression (on MNIST)

We test each method on progressively more complex architectures:

1. **(A) MLP on MNIST** - Smallest architecture, simplest baseline
2. **(B) MLP Mixer on MNIST** - Intermediate complexity
3. **(C) ConvNet on MNIST** - Most complex architecture

### Tokenization Study

A key research question is the effect of different tokenization strategies. We compare:

- **MDCT-based tokenization** - Using Modified Discrete Cosine Transform to convert input to frequency domain tokens (similar to MDCTCodec)
- **Reshape-based tokenization** - Using pixel-shuffle/pixel-unshuffle operations (via `einops.Rearrange`) to create tokens from spatial patches, similar to Vision Transformers (ViTs)

This comparison helps understand whether the frequency-domain representation provided by MDCT offers advantages over simple spatial reshaping for tokenization.

### Dataset Progression

After establishing baselines on MNIST, we extend to audio:

1. **(A) MNIST Data** - Initial validation and baseline establishment
2. **(B) Audio Data** - Application to target domain (with MDCT-encoded audio)

### Experimental Matrix

The complete experimental matrix covers:

- **4 Methods** × **3 Architectures** × **2 Tokenization Strategies** × **2 Datasets** = Systematic baseline establishment

This systematic approach ensures we understand:
- Performance characteristics of each method at each complexity level
- The impact of tokenization strategy (MDCT vs reshape)
- How findings generalize from MNIST to audio domain

## Reference Implementations

This project includes **trusted PyTorch reference implementations** in the [`references/`](references/) directory that serve as training wheels for debugging and triaging sources of errors. These implementations provide validated baselines for comparison when developing the JAX/Flax versions.

### Available References

- **`flow_matching_mnist.py`** - PyTorch implementation of conditional Flow Matching for MNIST
  - Validated Flow Matching implementation with class conditioning
  - MLP-based architecture with adaptive layer normalization
  - Useful for debugging flow matching objectives and sampling procedures

- **`mean_flow_mnist.py`** - PyTorch implementation of Mean Flow for MNIST
  - Validated Mean Flow implementation with corrected JVP handling
  - Includes fixes based on comparison with official implementations
  - Useful for debugging mean flow loss computation and fast sampling

These reference implementations can be used to:
- Validate loss computation correctness
- Compare sampling behavior and quality
- Debug architecture-specific issues
- Triage whether issues stem from JAX/Flax implementation vs. method design

See the [Reference Implementations README](references/README.md) for detailed documentation.

## Overview

This project builds an **Improved Mean Flow audio encoder** that operates in the MDCT domain. The systematic experimental plan ensures we validate each component before applying it to audio encoding tasks.

The implementation includes:

- **Improved Mean Flow (iMF) Audio Encoder**: Core architecture for audio encoding in MDCT domain
- **MNIST Benchmarking**: Initial validation of iMF on MNIST image generation
- **Conditional Flow Models**: Class-conditional generative models for both MNIST and audio
- **MDCT Preprocessing**: Modified Discrete Cosine Transform utilities for audio encoding
- **Baseline Flow Matching**: Standard Flow Matching implementation for comparison
- **Tokenizer Comparison**: MDCT vs reshape-based tokenization strategies
- **Evaluation Tools**: Metrics, sampling, and classifier-based evaluation

## Features

- 🎵 **Audio Encoder**: Improved Mean Flow encoder for audio in MDCT domain
- 📊 **MNIST Benchmarking**: Validation of iMF method on MNIST before audio application
- 🚀 **JAX/Flax Implementation**: High-performance, GPU-accelerated training
- 🔬 **Research Implementation**: Both baseline and improved Mean Flow objectives
- 🔄 **Tokenizer Comparison**: MDCT vs reshape-based tokenization study
- 🔍 **Reference Implementations**: Trusted PyTorch implementations for debugging and validation
- 🧪 **Comprehensive Testing**: Unit tests for core components including MDCT
- 📈 **Evaluation Pipeline**: Metrics, sampling, and classifier-based evaluation

## Quick Start

See the [Documentation](documentation/README.md) for comprehensive guides:

- **[Installation Guide](documentation/user_guide/installation.md)** - Setup instructions
- **[Quick Start Guide](documentation/user_guide/quick_start.md)** - Get started quickly
- **[Configuration Reference](documentation/user_guide/configuration.md)** - Configuration options
- **[Examples](documentation/user_guide/examples.md)** - Usage examples

## Project Structure

```
meanflow_audio_codec/
├── meanflow_audio_codec/          # Main package
│   ├── models/             # Flow models (for audio encoding) and classifiers
│   ├── trainers/           # Training loops and utilities
│   ├── evaluators/         # Sampling and metrics
│   ├── datasets/           # Data loading (MNIST for benchmarking)
│   ├── preprocessing/      # MDCT utilities for audio encoding
│   └── configs/           # Configuration dataclasses
├── test/                   # Unit tests (including MDCT tests)
├── meanflow_audio_codec/   # Main package
│   ├── tools/              # Utility scripts and benchmarks
│   │   └── benchmarks/     # Performance benchmark scripts
│   └── references/         # Trusted PyTorch reference implementations
├── documentation/          # Research notes and equations
│   ├── improved_meanflow_key_eqn.md  # iMF mathematical formulation
│   └── mdct/               # MDCT codec documentation
└── pyproject.toml          # Project configuration
```

## Documentation

Comprehensive documentation is available in the [`documentation/`](documentation/README.md) directory, organized into four main categories:

- **[User Guide](documentation/user_guide/README.md)** - Installation, quick start, configuration, and examples
- **[Developer Guide](documentation/developer_guide/README.md)** - Architecture, code organization, and API reference
- **[Research](documentation/research/README.md)** - Mathematical foundations (MeanFlow, Improved MeanFlow, MDCT)
- **[Implementation](documentation/implementation/README.md)** - Performance analysis, optimization, and benchmarks

## Testing

Run the test suite:

```bash
uv run pytest test/
```

Key test files:
- `test_improved_mean_flow.py`: Tests for Improved Mean Flow implementation
- `test_mdct.py`: MDCT forward/inverse transform and perfect reconstruction tests
- `test_mdct_perfect_reconstruction.py`: Round-trip MDCT reconstruction validation
- `test_mdct_reference.py`: Comparison with reference MDCT implementations
- `test_gelu.py`: GELU activation tests

## Research Background

This project applies Improved Mean Flows to audio encoding. The encoder operates entirely in the **MDCT (Modified Discrete Cosine Transform) domain**, following the approach of MDCTCodec.

The encoder pipeline: **Audio → MDCT → iMF Encoder → Latent → iMF Decoder → MDCT → Audio**

For detailed mathematical foundations, see the [Research Documentation](documentation/research/README.md):
- [MeanFlow Theory](documentation/research/meanflow/meanflow_key_eqn.md)
- [Improved MeanFlow Theory](documentation/research/improved_meanflow/improved_meanflow_key_eqn.md)
- [MDCT Theory](documentation/research/mdct/theory.md)
- [MDCTCodec Reference](documentation/research/mdct/mdctcodec_key_eqn.md)

## License

[Add your license here]

## Citation

If you use this code in your research, please cite the relevant papers:

### Core Methods

```bibtex
@article{geng2025meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J. Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}

@article{geng2024improved,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Ziyao and Lu, Chris and Wu, Yilun and Shechtman, Eli and Kolter, Zico and He, Jun-Yan},
  journal={arXiv preprint arXiv:2512.02012},
  year={2024}
}

@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@article{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K.},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

### Architecture References

```bibtex
@article{peebles2022dit,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2022}
}

@article{ma2024sit,
  title={SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers},
  author={Ma, Nanye and Goldstein, Mark and Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric and Xie, Saining},
  journal={arXiv preprint arXiv:2401.08740},
  year={2024}
}

@article{tolstikhin2021mlp,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Andreas and Steiner, Daniel and Keysers, Daniel and Uszkoreit, Jakob and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={24261--24272},
  year={2021}
}

@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11976--11986},
  year={2022}
}

@article{woo2023convnextv2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Xie, Saining and Liu, Zhuang and Zhang, Xiyang and Yuille, Alan L. and Koltun, Vladlen},
  journal={arXiv preprint arXiv:2301.00808},
  year={2023}
}
```

### Audio Codec References

```bibtex
@article{jiang2024mdctcodec,
  title={MDCTCodec: A Lightweight MDCT-Based Neural Audio Codec Towards High Sampling Rate and Low Bitrate Scenarios},
  author={Jiang, Yifan and Ai, Yuxuan and Zheng, Ziyue and Du, Yuxin and Lu, Yuping and Ling, Zhenhua},
  journal={arXiv preprint arXiv:2411.00464},
  year={2024}
}

@article{defossez2022encodec,
  title={High Fidelity Neural Audio Compression},
  author={D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}

@misc{stability2024openaudiocodec,
  title={Stability AI Open Audio Codec},
  author={Stability AI},
  howpublished={\url{https://github.com/Stability-AI/stable-audio-codec}},
  year={2024},
  note={Open source audio codec implementation}
}

@book{pohlmann2010principles,
  title={Principles of Digital Audio},
  author={Pohlmann, Ken C.},
  edition={6th},
  publisher={McGraw-Hill Professional},
  year={2010},
  note={Comprehensive reference on digital audio, ADC/DAC principles}
}
```

## Contributing

See the [Contributing Guide](documentation/developer_guide/contributing.md) for guidelines on contributing to the project.
