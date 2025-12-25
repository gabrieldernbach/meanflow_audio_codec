# Implementation Gap Analysis

## Executive Summary

**Plan**: 4 methods × 3 architectures × 2 tokenizations × 2 datasets = **48 configs**

**Current State**: 
- ✅ All methods, architectures, datasets, and tokenization code implemented
- ❌ **Tokenization NOT integrated into training pipeline** (critical blocker)
- ❌ Only 24 configs exist (missing tokenization variants)
- ❌ Configs don't specify tokenization strategy explicitly

**Critical Blocker**: Tokenization must be integrated into training pipeline before tokenization comparison experiments can be conducted.

## Research Plan Overview

According to the README, the experimental matrix should cover:
- **4 Methods**: autoencoder, flow_matching, mean_flow, improved_mean_flow
- **3 Architectures**: mlp, mlp_mixer, convnet  
- **2 Tokenization Strategies**: MDCT-based, reshape-based
- **2 Datasets**: mnist, audio

**Expected Total**: 4 × 3 × 2 × 2 = **48 configs**

## Current State

### Configs Available
- **24 configs exist** (4 methods × 3 architectures × 2 datasets)
- Configs follow pattern: `method={method}--architecture={arch}--dataset={dataset}.json`
- Tokenization strategy is **NOT** explicitly specified in:
  - Config filenames
  - Config JSON content (no `tokenization_strategy` field)
- Tokenization defaults are set in code (`config.py`):
  - MNIST → `reshape` (default in `MNISTConfig.__post_init__`)
  - Audio → `mdct` (default in `AudioConfig.__post_init__`)
- **Note**: `tools/generate_configs.py` script exists and would generate 48 configs with tokenization in filename, but it hasn't been run to create the full set

### Implementation Status

#### ✅ Fully Implemented
1. **Methods** (4/4):
   - `autoencoder` - via loss_strategy inference
   - `flow_matching` - `FlowMatchingLoss` class
   - `mean_flow` - `MeanFlowLoss` class  
   - `improved_mean_flow` - `ImprovedMeanFlowLoss` class

2. **Architectures** (3/3):
   - `mlp` - `mlp_flow.py`
   - `mlp_mixer` - `mlp_mixer.py`
   - `convnet` - `conv_flow.py`, `simple_conv_flow.py`

3. **Datasets** (2/2):
   - `mnist` - `datasets/mnist.py`
   - `audio` - `datasets/audio.py`

4. **Tokenization Strategies** (2/2):
   - `mdct` - `preprocessing/tokenization.py` (`MDCTTokenization`)
   - `reshape` - `preprocessing/tokenization.py` (`ReshapeTokenization`)

#### ❌ Critical Gap: Tokenization Not Integrated

**Tokenization is implemented but NOT used in the training pipeline.**

Evidence:
- `trainers/train.py` - No tokenization usage
- `trainers/training_steps.py` - No tokenization usage
- `train.py` - No tokenization usage
- Data flows directly from dataset → model without tokenization step

**Impact**: The tokenization study (MDCT vs reshape) cannot be conducted with current implementation.

### Missing Configs

The following 24 configs are missing (explicit tokenization variants):

**MNIST with MDCT tokenization** (12 configs):
- `method=autoencoder--architecture=mlp--dataset=mnist--tokenization=mdct.json`
- `method=autoencoder--architecture=mlp_mixer--dataset=mnist--tokenization=mdct.json`
- `method=autoencoder--architecture=convnet--dataset=mnist--tokenization=mdct.json`
- `method=flow_matching--architecture=mlp--dataset=mnist--tokenization=mdct.json`
- `method=flow_matching--architecture=mlp_mixer--dataset=mnist--tokenization=mdct.json`
- `method=flow_matching--architecture=convnet--dataset=mnist--tokenization=mdct.json`
- `method=mean_flow--architecture=mlp--dataset=mnist--tokenization=mdct.json`
- `method=mean_flow--architecture=mlp_mixer--dataset=mnist--tokenization=mdct.json`
- `method=mean_flow--architecture=convnet--dataset=mnist--tokenization=mdct.json`
- `method=improved_mean_flow--architecture=mlp--dataset=mnist--tokenization=mdct.json`
- `method=improved_mean_flow--architecture=mlp_mixer--dataset=mnist--tokenization=mdct.json`
- `method=improved_mean_flow--architecture=convnet--dataset=mnist--tokenization=mdct.json`

**Audio with reshape tokenization** (12 configs):
- `method=autoencoder--architecture=mlp--dataset=audio--tokenization=reshape.json`
- `method=autoencoder--architecture=mlp_mixer--dataset=audio--tokenization=reshape.json`
- `method=autoencoder--architecture=convnet--dataset=audio--tokenization=reshape.json`
- `method=flow_matching--architecture=mlp--dataset=audio--tokenization=reshape.json`
- `method=flow_matching--architecture=mlp_mixer--dataset=audio--tokenization=reshape.json`
- `method=flow_matching--architecture=convnet--dataset=audio--tokenization=reshape.json`
- `method=mean_flow--architecture=mlp--dataset=audio--tokenization=reshape.json`
- `method=mean_flow--architecture=mlp_mixer--dataset=audio--tokenization=reshape.json`
- `method=mean_flow--architecture=convnet--dataset=audio--tokenization=reshape.json`
- `method=improved_mean_flow--architecture=mlp--dataset=audio--tokenization=reshape.json`
- `method=improved_mean_flow--architecture=mlp_mixer--dataset=audio--tokenization=reshape.json`
- `method=improved_mean_flow--architecture=convnet--dataset=audio--tokenization=reshape.json`

## Required Work

### 1. Integrate Tokenization into Training Pipeline (HIGH PRIORITY)

**Location**: `meanflow_audio_codec/trainers/train.py`

**Changes needed**:
1. Load tokenization strategy from config
2. Apply tokenization to data before passing to model
3. Apply detokenization after model output (for evaluation/sampling)
4. Update model input/output dimensions based on tokenization

**Key considerations**:
- Tokenization changes data shape: `[B, ...]` → `[B, n_tokens, token_dim]`
- Model architecture must handle tokenized input
- Loss computation must work with tokenized representations
- Sampling/evaluation must detokenize back to original format

### 2. Generate Missing Configs

**Location**: `meanflow_audio_codec/tools/generate_configs.py`

**Status**: Script exists and generates 48 configs with tokenization in filename, but:
- Script has not been run to generate the full set
- Current 24 configs in `configs/` directory don't include tokenization
- Need to run: `uv run python -m meanflow_audio_codec.tools.generate_configs --output-dir configs --base-only`

**Action**: Run config generation script to create all 48 configs

### 3. Update Model Architectures (if needed)

**Check**: Do current architectures handle tokenized input correctly?
- MLP: May need to flatten tokens
- MLP Mixer: Should handle tokens naturally
- ConvNet: May need spatial reshaping

### 4. Update Data Loading

**Location**: `meanflow_audio_codec/datasets/`

**Changes needed**:
- Optionally apply tokenization in data loader
- Or apply in training loop (preferred for flexibility)

## Summary

| Component | Status | Gap |
|-----------|--------|-----|
| Methods | ✅ Complete | None |
| Architectures | ✅ Complete | None |
| Datasets | ✅ Complete | None |
| Tokenization (code) | ✅ Complete | **Not integrated into pipeline** |
| Tokenization (configs) | ❌ Missing | 24 configs missing |
| Training integration | ❌ Missing | Tokenization not used |

**Critical Blocker**: Tokenization must be integrated into the training pipeline before any tokenization comparison experiments can be run.

