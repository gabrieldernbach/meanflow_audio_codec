# Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd meanflow-audio-codec
```

2. Install dependencies:
```bash
uv sync
```

3. (Optional) For Apple Silicon with Metal support:
```bash
uv sync --extra metal
```

4. (Optional) For fast MP3 audio loading with minimp3py:
```bash
# Requires C compiler (gcc/clang) and Python development headers
# On Linux: sudo apt-get install build-essential python3-dev
# On macOS: xcode-select --install (if not already installed)
uv sync --extra audio
```

**Note**: The `audio` extra installs `minimp3py` which requires compilation. For optimal performance, install with:
```bash
CFLAGS='-O3 -march=native' uv pip install git+https://github.com/f0k/minimp3py.git
```

The audio loader will fall back to librosa if minimp3py is not available, but minimp3py provides significantly faster MP3 decoding.

## Dependencies

### Core Dependencies
- `jax` & `jaxlib`: Numerical computing and automatic differentiation
- `flax`: Neural network library
- `optax`: Optimization library
- `tensorflow-datasets`: Dataset loading
- `matplotlib`: Visualization
- `numpy`: Numerical operations
- `librosa`: Audio loading (fallback if minimp3py not available)

### Optional Dependencies

Install with `uv sync --extra <name>`:

- **`audio`**: Fast MP3 loading with `minimp3py` (requires C compiler)
  - Requires: C compiler (gcc/clang) and Python development headers
  - Provides: 10-15x faster MP3 decoding compared to librosa
  - Falls back to librosa if not installed

- **`metal`**: JAX Metal support for Apple Silicon (macOS only)
  - Automatically enabled on Apple Silicon if available

- **`all`**: All optional dependencies
  ```bash
  uv sync --extra all
  ```

See `pyproject.toml` for the complete list.

