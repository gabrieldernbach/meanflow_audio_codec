# Benchmarks

This directory contains benchmark scripts for various components of the meanflow_audio_codec project.

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

### Torch Comparison (Potentially Outdated)
The following benchmarks are still in `test/` directory but may be outdated. They require external `flowmo` repository at `~/git/flowmo`:
- **`test/benchmark_audio_vs_torch.py`** - Compares JAX implementation with PyTorch flowmo implementation.
- **`test/benchmark_torch_only.py`** - Standalone torch benchmark.

These torch benchmarks may still be useful if you have the flowmo repository, but are not actively maintained.

## Usage

Most benchmarks can be run directly with `uv run`:

```bash
# MDCT benchmarks
uv run benchmarks/benchmark_mdct.py
uv run benchmarks/benchmark_mdct_simple.py

# Audio benchmarks
uv run benchmarks/benchmark_audio_simple_test.py
uv run benchmarks/benchmark_prefetch.py

# MP3 loader comparison (requires data directory)
uv run benchmarks/benchmark_mp3_loaders.py --data-dir /path/to/mp3/files
```

## Notes

- Benchmarks that require audio data will look for files in the directory specified by `AUDIO_DATA_DIR` environment variable, or default to `~/datasets/wavegen`
- Some benchmarks may require optional dependencies (e.g., `minimp3py`, `np3`, `pymp3`) which are gracefully handled if not available

