# Benchmarks

This document describes the benchmark scripts available in `meanflow_audio_codec/tools/benchmarks/`.

All benchmarks can be run as Python modules:
```bash
python -m meanflow_audio_codec.tools.benchmarks.benchmark_name
# Or with uv:
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_name
```

## Current Benchmarks

### Audio Processing
- **`benchmark_audio_simple_test.py`** - Simple benchmark for audio dataloader components (glob, load, shuffle, batch)
- **`benchmark_prefetch.py`** - Benchmark comparing prefetch vs no-prefetch for audio loading
- **`benchmark_mp3_loaders.py`** - Comparison of different MP3 loading libraries (librosa, np3, pymp3, minimp3py)

### MDCT (Modified Discrete Cosine Transform)
- **`benchmark_mdct.py`** - Comprehensive MDCT benchmark comparing JAX Metal, JAX CPU, NumPy baseline, and FFT variants
- **`benchmark_mdct_simple.py`** - Simpler MDCT benchmark comparing JAX Metal vs NumPy baseline vs Reference implementation

## Outdated Benchmarks (Removed)

The following outdated benchmarks have been removed:
- **`benchmark_audio_dataloaders.py`** - Referenced non-existent functions `load_audio_train` and `load_audio_train_grain`. The current API uses `build_audio_pipeline` instead.
- **`benchmark_audio_quick.py`** - Referenced non-existent functions `load_audio_train` and `load_audio_train_grain`.

### Torch Comparison
The following benchmarks compare JAX implementation with PyTorch flowmo implementation. They require external `flowmo` repository at `~/git/flowmo`:
- **`benchmark_audio_vs_torch.py`** - Compares JAX implementation with PyTorch flowmo implementation
- **`benchmark_torch_only.py`** - Standalone torch benchmark

These benchmarks can be run as:
```bash
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_audio_vs_torch
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_torch_only
```

### Method Comparison
- **`benchmark_meanflow_vs_improved.py`** - Compares original MeanFlow vs Improved MeanFlow on MNIST

```bash
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_meanflow_vs_improved
```

### Audio Loader Benchmarks
- **`benchmark_audio_loader.py`** - Benchmark audio dataloader implementations and identify bottlenecks

```bash
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_audio_loader
```

## Usage

Most benchmarks can be run directly with `uv run`:

```bash
# MDCT benchmarks
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct_simple

# Audio benchmarks
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_audio_simple_test
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_prefetch

# MP3 loader comparison (requires data directory)
uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_mp3_loaders --data-dir /path/to/mp3/files
```

## Notes

- Benchmarks that require audio data will look for files in the directory specified by `AUDIO_DATA_DIR` environment variable, or default to `~/datasets/wavegen`
- Some benchmarks may require optional dependencies (e.g., `minimp3py`, `np3`, `pymp3`) which are gracefully handled if not available

