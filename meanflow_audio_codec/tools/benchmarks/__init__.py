"""Benchmark utilities for performance testing.

This module contains benchmark scripts for testing performance of various components:
- MDCT implementations (benchmark_mdct.py, benchmark_mdct_simple.py)
- Audio loading pipelines (benchmark_audio_simple_test.py, benchmark_audio_loader.py, benchmark_prefetch.py)
- MP3 loader comparison (benchmark_mp3_loaders.py)
- Torch comparison (benchmark_audio_vs_torch.py, benchmark_torch_only.py)
- Method comparison (benchmark_meanflow_vs_improved.py)

All benchmarks follow the standard entry point pattern with a `main()` function.

Benchmarks can be run as modules:
    python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct

Or with uv:
    uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct

See documentation/implementation/benchmarks/README.md for detailed usage information.
"""

__all__ = []

