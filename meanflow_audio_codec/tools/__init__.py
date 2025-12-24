"""Tools for dataset downloading, preparation, and benchmarking.

This module contains utility scripts organized into submodules:
- Dataset preparation tools (download_wavegen, etc.)
- Benchmark scripts (benchmarks/benchmark_*.py)
- Evaluation utilities (evaluate_all, aggregate_results, etc.)
- Configuration generation (generate_configs, generate_tables)

These tools are separate from the generic data loading code in meanflow_audio_codec.datasets.

Following BigVision's structure (big_vision.tools), tools are organized as a Python
module within the package, allowing them to be imported and executed as:

    # Run as module
    python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]
    python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct
    
    # Or with uv
    uv run python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]
    uv run python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct
    
    # Or import programmatically
    from meanflow_audio_codec.tools import download_wavegen
    download_wavegen.main()
"""

__all__ = [
    "download_wavegen",
]

