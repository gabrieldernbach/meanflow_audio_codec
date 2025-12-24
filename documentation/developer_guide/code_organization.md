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

