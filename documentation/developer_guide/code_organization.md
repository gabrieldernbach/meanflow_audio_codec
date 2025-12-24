# Code Organization

This document describes the code organization patterns used in the Meanflow Audio Codec project, which follows patterns inspired by Google Research's BigVision repository.

## Repository Structure

### Current Structure

```
meanflow_audio_codec/
├── meanflow_audio_codec/        # Main package
│   ├── configs/                 # Configuration dataclasses
│   ├── models/                  # Model definitions
│   ├── trainers/                # Training loops and utilities
│   ├── evaluators/              # Evaluation utilities
│   ├── datasets/                # Data loading
│   ├── preprocessing/           # Preprocessing (MDCT)
│   ├── tools/                   # Dataset preparation tools
│   └── proj/                    # Project-specific code
├── test/                        # Test suite
├── documentation/               # Documentation
└── configs/                     # Example configuration files
```

### Comparison with BigVision

| Aspect | BigVision | meanflow_audio_codec | Status |
|--------|-----------|---------------------|--------|
| **Entry Point** | Root-level `train.py` | Package-based | ⚠️ Different pattern |
| **Package Structure** | Flat or minimal nesting | Nested package | ⚠️ More nested |
| **Project Code** | `proj/project_name/` | `proj/` exists | ⚠️ Not fully utilized |
| **Component Organization** | Modular subdirectories | ✅ Modular subdirectories | ✅ Aligned |

## Component Organization

### Models

Models are organized in `models/` with consistent APIs:
- `models/mlp_flow.py`: MLP-based flow models
- `models/conv_flow.py`: Convolutional flow models
- `models/train_state.py`: Custom TrainState
- `models/__init__.py`: Clean exports

**Status**: ✅ Aligned with BigVision patterns

### Trainers

Training logic is well-separated:
- `trainers/train.py`: Main training loop
- `trainers/training_steps.py`: Individual step functions
- `trainers/utils.py`: Shared utilities

**Status**: ✅ Good separation

### Evaluators

Evaluation utilities in `evaluators/`:
- `evaluators/sampling.py`: Sampling functions
- `evaluators/metrics.py`: Metrics computation

**Status**: ✅ Aligned

### Datasets

Data loading in `datasets/`:
- `datasets/mnist.py`: MNIST loading
- `datasets/audio.py`: Audio loading

**Status**: ✅ Aligned

## Tools Organization (BigVision Pattern)

### Structure

Tools are organized as a **Python module within the package**:

```
meanflow_audio_codec/
├── tools/
│   ├── __init__.py
│   └── download_wavegen.py
├── datasets/              # Generic dataset loaders
└── ...
```

### Key Characteristics

1. **Module-based**: Tools are Python modules, not standalone scripts
2. **Importable**: Can be imported as `from meanflow_audio_codec.tools import download_wavegen`
3. **Runnable**: Can be executed as `python -m meanflow_audio_codec.tools.download_wavegen`
4. **Separated from datasets**: Tools are separate from generic dataset loaders

### Usage

**Module execution:**
```bash
# Run tool as Python module
python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]

# Or with uv
uv run python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]
```

**Import pattern:**
```python
from meanflow_audio_codec.tools import download_wavegen
download_wavegen.main()
```

## Dataset Organization: Three-Layer Separation

Following BigVision's pattern, we separate dataset concerns into three layers:

### 1. Dataset Specification
**What dataset to use**

- Defined in configuration files or dataset builders
- Specifies dataset name, source, splits, etc.
- Example: `"mnist"`, `"WaveGenAI/youtube-cc-by-music"`, or a custom dataset path

### 2. Multi-process Loading
**How data is loaded during training**

- Handled by generic data loading code in `meanflow_audio_codec/datasets/`
- Supports multi-process loading, prefetching, shuffling, batching
- Works with any dataset that follows the expected format
- Examples:
  - `audio.py`: Generic audio loader (works with any audio directory)
  - `mnist.py`: Generic MNIST loader (works with TFDS)

### 3. Downloading/Preparation
**Scripts that download/prepare datasets**

- Located in `meanflow_audio_codec/tools/` module
- Dataset-specific download/preparation logic
- Python modules that can be run with `python -m` or imported
- Examples:
  - `meanflow_audio_codec/tools/download_wavegen.py`: Downloads WaveGen dataset

### Example: Audio Datasets

**Generic Loader (`audio.py`):**
```python
# This works with ANY audio directory - wavegen, custom collection, etc.
from meanflow_audio_codec.datasets.audio import build_audio_pipeline

# Load from WaveGen (after downloading)
pipeline = build_audio_pipeline(data_dir="~/datasets/wavegen", ...)

# Load from custom collection
pipeline = build_audio_pipeline(data_dir="~/datasets/custom_audio", ...)
```

**Download Script:**
```bash
# Download WaveGen dataset (BigVision-style module execution)
python -m meanflow_audio_codec.tools.download_wavegen --output-dir ~/datasets/wavegen
```

**Configuration:**
```python
# In config or training script
config = TrainFlowConfig(
    data_dir="~/datasets/wavegen",  # Dataset specification
    # ... other config
)
```

### Benefits

1. **Separation of Concerns**: Download logic separate from loading logic
2. **Reusability**: `audio.py` can load many different audio datasets
3. **Maintainability**: Easy to add new download scripts without touching generic loaders
4. **Clarity**: Clear distinction between "what to download" vs "how to load"

## Code Style & Patterns

### Configuration System

**Current Pattern:**
- Dataclass configs: `TrainFlowConfig`, `TrainClassifierConfig`, etc. in `configs/config.py`
- Explicit parameters: All config fields are explicit dataclass fields
- Type-safe with dataclasses
- IDE-friendly (autocomplete, type checking)

**Example:**
```python
config = TrainFlowConfig(
    batch_size=128,
    n_steps=10000,
    output_dir=Path("./outputs"),
    run_name="my_experiment",
    # ... many explicit fields
)
train_flow(config)
```

**Potential Improvements:**
- Consider config file loading (YAML/JSON) for easier experimentation
- Add `workdir` pattern for standardized checkpoint/resume behavior
- Support config inheritance/merging for common defaults

### Error Handling

**Current Pattern:**
- Generally explicit, but some defensive fallbacks exist
- Recommendation: Follow repo rules - raise explicit errors instead of silent defaults

**Example (to avoid):**
```python
# ❌ Silent fallback (violates repo rules)
data_dir = config.data_dir or str(Path.home() / "datasets" / "mnist")
```

**Recommended:**
```python
# ✅ Explicit error
if config.data_dir is None:
    raise ValueError("data_dir must be provided")
```

## Module Organization Best Practices

### MDCT Module Example

The MDCT module (`preprocessing/mdct.py`) demonstrates good practices but could benefit from:

1. **Module-level docstring**: Explain the module's purpose
2. **Public API first**: Place public functions (`mdct()`, `imdct()`) near the top
3. **Testing utilities separate**: Move baseline implementations to test files
4. **Comprehensive docstrings**: Include Args/Returns/Examples sections

### Recommended Module Structure

1. **Imports** (grouped: stdlib, third-party, local)
2. **Constants/Configuration** (at top)
3. **Type definitions** (if using type hints extensively)
4. **Public API** (main entry points first)
5. **Core implementations** (internal functions)
6. **Utilities** (helper functions)
7. **Private helpers** (prefixed with `_`)

## Recommendations

### High Priority

1. **Create root-level entry point**
   - Add `train.py` at repository root with `main()` function
   - Update `pyproject.toml` entry point to reference it
   - Make it config-driven using existing `TrainFlowConfig`

2. **Move standalone scripts to `proj/`**
   - Move trial scripts to `meanflow_audio_codec/proj/` subdirectories
   - Create project-specific configs in `proj/` subdirectories

3. **Implement JSON logging**
   - Add structured JSON log file (e.g., `metrics.jsonl` or `train_log.json`)
   - Log step, loss, learning rate, etc. as JSON objects
   - Keep stdout logging for human readability

4. **Add checkpoint resume**
   - Implement automatic resume from checkpoint if workdir contains checkpoint
   - Add `--resume` flag or auto-detect from workdir

### Medium Priority

5. **Standardize on `workdir` pattern**
   - Replace `output_dir/run_name` with single `workdir` parameter
   - Store all outputs (checkpoints, samples, logs) in workdir

6. **Add config file support**
   - Support loading configs from YAML/JSON files
   - Keep dataclass system but add file loading
   - Allow config inheritance/merging

7. **Fix defensive coding violations**
   - Remove silent fallbacks (e.g., data_dir default path)
   - Raise explicit errors when required values are missing

### Low Priority

8. **Consider flattening package structure**
   - Current: `meanflow_audio_codec/meanflow_audio_codec/`
   - Could be: `meanflow_audio_codec/` (if package name allows)

9. **Add CLI argument parsing**
   - Use argparse or click for command-line interface
   - Support both config files and CLI overrides

## Summary

The `meanflow_audio_codec` repository follows many BigVision patterns and has good modular organization. The main areas for improvement are:

1. **Entry point structure**: Add root-level `train.py`
2. **Project organization**: Move standalone scripts to `proj/`
3. **Logging**: Implement structured JSON logging
4. **Checkpointing**: Add automatic resume functionality
5. **Workdir pattern**: Standardize on single workdir instead of `output_dir/run_name`

The codebase is well-structured and maintainable. These changes would bring it closer to BigVision conventions while maintaining its current strengths (type-safe configs, clear organization, good separation of concerns).

## Entry Point Patterns

### Standard Entry Point Pattern

All executable scripts should follow a consistent entry point pattern:

```python
def main():
    """Main entry point for [script purpose]."""
    # Script logic here
    pass


if __name__ == "__main__":
    main()
```

### Key Principles

1. **Always define `main()` function**: All executable scripts should have a `main()` function that contains the script logic
2. **Use `if __name__ == "__main__"` guard**: Always use the standard guard to call `main()`
3. **Document the purpose**: Include a docstring explaining what the script does
4. **Module-based execution**: Scripts in `tools/` should be executable as modules:
   ```bash
   python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct
   uv run python -m meanflow_audio_codec.tools.download_wavegen --output-dir ~/datasets
   ```

### Examples

**Good:**
```python
def main():
    """Main entry point for MDCT benchmark."""
    print("Running benchmark...")
    run_benchmarks()


if __name__ == "__main__":
    main()
```

**Bad:**
```python
# ❌ Code directly in if __name__ block
if __name__ == "__main__":
    print("Running benchmark...")
    run_benchmarks()
```

## Script Organization Guidelines

### Directory Structure

Scripts are organized into three main categories:

1. **Utility Scripts** (`meanflow_audio_codec/tools/`):
   - Dataset preparation tools (`download_wavegen.py`)
   - Benchmark scripts (`tools/benchmarks/benchmark_*.py`)
   - Evaluation utilities (`evaluate_all.py`, `aggregate_results.py`)
   - Configuration generation (`generate_configs.py`)
   - All general-purpose utility scripts

2. **Project-Specific Scripts** (`meanflow_audio_codec/proj/`):
   - Experimental training scripts
   - Project-specific configurations
   - Trial implementations
   - Examples: `audio_autoencoder/train.py`, `mnist_trial/train.py`

3. **Main Entry Points** (root level):
   - Primary training script (`train.py`)
   - Should be minimal and delegate to package code

### When to Use Each Location

- **Use `tools/`** for:
  - Reusable utilities that work across projects
  - Benchmark scripts
  - Data preparation scripts
  - General-purpose evaluation tools

- **Use `proj/`** for:
  - Experimental code
  - Project-specific training scripts
  - Trial implementations
  - Code that may not be maintained long-term

- **Use root level** for:
  - Primary entry points (e.g., `train.py`)
  - Only the most important, stable scripts

### Benchmarks Organization

All benchmark scripts are located in `meanflow_audio_codec/tools/benchmarks/`:

- `benchmark_mdct.py` - MDCT implementation benchmarks
- `benchmark_audio_loader.py` - Audio dataloader benchmarks
- `benchmark_meanflow_vs_improved.py` - Method comparison benchmarks
- And other performance testing scripts

Benchmarks can be run as modules:
```bash
python -m meanflow_audio_codec.tools.benchmarks.benchmark_mdct
```

## Naming Conventions

### File and Module Names

- **Script files**: `snake_case.py` (e.g., `train.py`, `benchmark_mdct.py`)
- **Module files**: `snake_case.py` (e.g., `audio.py`, `mdct.py`)
- **Package directories**: `snake_case/` (e.g., `datasets/`, `preprocessing/`)

### Class Names

- **Classes**: `PascalCase` (e.g., `ConditionalFlow`, `TrainState`, `MDCTConfig`)

### Function Names

- **Functions**: `snake_case` (e.g., `train_flow`, `build_audio_pipeline`, `mdct`)

### Constants

- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`, `MAX_ITERATIONS`)

### Private Members

- **Private functions/classes**: Prefix with single underscore `_` (e.g., `_load_audio`, `_mdct_direct`)
- **Module-private**: Use `__all__` to control public API

### Examples

```python
# ✅ Good naming
class ConditionalFlow:
    def train_step(self, batch):
        pass

DEFAULT_BATCH_SIZE = 32

def build_audio_pipeline(data_dir: str):
    pass

def _internal_helper():
    pass

# ❌ Bad naming
class conditional_flow:  # Should be PascalCase
    def TrainStep(self, batch):  # Should be snake_case
        pass

defaultBatchSize = 32  # Should be UPPER_SNAKE_CASE
```

## Module Export Guidelines

### Public API Definition

Each module should clearly define its public API using `__all__`:

```python
from meanflow_audio_codec.models.mlp_flow import ConditionalFlow, MLP

__all__ = [
    "ConditionalFlow",
    "MLP",
]
```

### Export Principles

1. **Explicit exports**: Always use `__all__` to define public API
2. **Import organization**: Group imports (stdlib, third-party, local)
3. **Version info**: Main package `__init__.py` should export `__version__`
4. **Documentation**: Include module-level docstrings explaining purpose

### Module Structure

```python
"""Module-level docstring explaining purpose and usage."""

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from meanflow_audio_codec.models import ConditionalFlow

# Public API
__all__ = [
    "ConditionalFlow",
]

# Implementation...
```

### Testing Utilities

Testing utilities (functions prefixed with `_` or marked for testing) should be:
- Clearly documented as testing utilities
- Included in `__all__` only if needed for external testing
- Examples: `_mdct_direct`, `mdct_fft` (marked with comments in `__all__`)

## Separation of Library vs Application Code

### Library Code

**Location**: `meanflow_audio_codec/` package modules

**Characteristics**:
- Reusable, well-tested components
- Stable APIs with `__all__` exports
- Comprehensive documentation
- Examples: `models/`, `datasets/`, `preprocessing/`, `trainers/`

### Application Code

**Location**: `tools/`, `proj/`, root-level scripts

**Characteristics**:
- Scripts that use library code
- May be experimental or project-specific
- Can have more flexible APIs
- Examples: `tools/download_wavegen.py`, `proj/audio_autoencoder/train.py`

### Reference Implementations

**Location**: `meanflow_audio_codec/references/`

**Purpose**:
- PyTorch reference implementations for debugging
- Trusted baselines for comparison
- Not part of the main library API
- Clearly marked as reference/debug code

### Guidelines

1. **Library code should not depend on application code**
2. **Application code imports and uses library code**
3. **Reference implementations are separate and clearly marked**
4. **Tools are separate modules, not part of core library**

## Repository Organization Patterns (Stability AI Inspired)

### Key Principles

1. **Clear module boundaries**: Each module has a single, well-defined purpose
2. **Consistent entry points**: All scripts follow the same `main()` pattern
3. **Organized utilities**: Tools and benchmarks are clearly categorized
4. **Separation of concerns**: Library code, application code, and references are distinct

### Directory Organization

```
meanflow_audio_codec/
├── meanflow_audio_codec/          # Main library package
│   ├── models/                    # Model definitions (library)
│   ├── trainers/                  # Training logic (library)
│   ├── datasets/                  # Data loading (library)
│   ├── preprocessing/             # Preprocessing (library)
│   ├── evaluators/                # Evaluation (library)
│   ├── configs/                   # Configuration (library)
│   ├── tools/                     # Utility scripts (application)
│   │   ├── benchmarks/            # Benchmark scripts
│   │   └── ...                    # Other utilities
│   ├── proj/                      # Project-specific code (application)
│   └── references/                # Reference implementations (debug)
├── test/                          # Test suite
├── documentation/                 # Documentation
├── configs/                       # Example configs
└── train.py                       # Main entry point
```

### Adding New Components

**Adding a new tool:**
1. Create script in `meanflow_audio_codec/tools/`
2. Follow entry point pattern with `main()` function
3. Add to `tools/__init__.py` if needed for imports
4. Document usage in module docstring

**Adding a new benchmark:**
1. Create script in `meanflow_audio_codec/tools/benchmarks/`
2. Name it `benchmark_*.py`
3. Follow entry point pattern
4. Document what it benchmarks

**Adding a new project:**
1. Create subdirectory in `meanflow_audio_codec/proj/`
2. Add project-specific scripts
3. Document project purpose in README or docstrings

**Adding library code:**
1. Add to appropriate module (`models/`, `datasets/`, etc.)
2. Export via `__init__.py` with `__all__`
3. Add comprehensive documentation
4. Write tests in `test/`

