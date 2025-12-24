"""Tools for dataset downloading and preparation.

This module contains utility scripts for downloading and preparing datasets.
These tools are separate from the generic data loading code in meanflow_audio_codec.datasets.

Following BigVision's structure (big_vision.tools), tools are organized as a Python
module within the package, allowing them to be imported and executed as:

    # Run as module
    python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]
    
    # Or with uv
    uv run python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]
    
    # Or import programmatically
    from meanflow_audio_codec.tools import download_wavegen
    download_wavegen.main()
"""

__all__ = [
    "download_wavegen",
]

