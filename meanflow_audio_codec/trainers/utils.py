"""Shared utilities for training, evaluation, and checkpointing."""

import csv
import hashlib
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import serialization
from flax.training import train_state as flax_train_state

from meanflow_audio_codec.configs.config import TrainFlowConfig
from meanflow_audio_codec.evaluators.performance import count_parameters
from meanflow_audio_codec.models import ConditionalFlow, TrainState


def save_json(path: Path, payload: dict) -> None:
    """Save a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> dict:
    """Load dict from JSON file. Returns empty dict if file doesn't exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    path: Path, state: TrainState | flax_train_state.TrainState
) -> None:
    """Save a TrainState checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path: Path, state_template: TrainState) -> TrainState:
    """Load a TrainState checkpoint from disk."""
    with path.open("rb") as f:
        data = f.read()
    return serialization.from_bytes(state_template, data)


def get_checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    """Get metadata JSON path for a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., step_00100.msgpack)
        
    Returns:
        Path to metadata file (e.g., step_00100.json)
    """
    return checkpoint_path.with_suffix(".json")


def get_checkpoint_size(checkpoint_path: Path) -> int:
    """Get checkpoint file size in bytes.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    if not checkpoint_path.exists():
        return 0
    return checkpoint_path.stat().st_size


def get_param_shapes(params: Any) -> dict[str, list[int]]:
    """Extract key parameter shapes for validation.
    
    Args:
        params: Model parameters (Flax params dict)
        
    Returns:
        Dictionary mapping parameter paths to shapes
    """
    shapes: dict[str, list[int]] = {}
    
    def extract_shapes_recursive(path: tuple, node: Any) -> None:
        """Recursively extract shapes from parameter tree."""
        if isinstance(node, dict):
            for key, value in node.items():
                extract_shapes_recursive(path + (key,), value)
        elif isinstance(node, (jnp.ndarray, np.ndarray)):
            param_path = "/".join(str(p) for p in path)
            shapes[param_path] = list(node.shape)
    
    extract_shapes_recursive((), params)
    return shapes


def save_checkpoint_metadata(
    checkpoint_path: Path,
    step: int,
    state: TrainState | flax_train_state.TrainState,
    config: TrainFlowConfig | None = None,
) -> None:
    """Save checkpoint metadata alongside checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        step: Training step number
        state: Training state
        config: Training configuration (optional)
    """
    metadata_path = get_checkpoint_metadata_path(checkpoint_path)
    
    # Collect metadata
    timestamp = datetime.now().isoformat()
    git_commit, git_commit_short = get_git_commit_hash()
    
    config_hash = None
    if config:
        config_dict = config.to_dict()
        config_hash = compute_config_hash(config_dict)
    
    # System info
    python_version = sys.version.split()[0]
    try:
        jax_version = jax.__version__
    except (ImportError, AttributeError):
        jax_version = None
    
    try:
        import flax
        flax_version = flax.__version__
    except (ImportError, AttributeError):
        flax_version = None
    
    platform_info = f"{platform.system()} {platform.release()}"
    
    # Model info
    param_info = count_parameters(state.params)
    param_shapes = get_param_shapes(state.params)
    
    # Checkpoint size
    checkpoint_size = get_checkpoint_size(checkpoint_path)
    
    metadata = {
        "step": step,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "git_commit_short": git_commit_short,
        "system_info": {
            "platform": platform_info,
            "python_version": python_version,
            "jax_version": jax_version,
            "flax_version": flax_version,
        },
        "checkpoint_size_bytes": checkpoint_size,
        "model_info": {
            "param_count": param_info["total"],
            "param_shapes": param_shapes,
        },
    }
    
    save_json(metadata_path, metadata)


def load_checkpoint_metadata(checkpoint_path: Path) -> dict | None:
    """Load checkpoint metadata from JSON file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Metadata dictionary, or None if metadata doesn't exist
    """
    metadata_path = get_checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return None
    return load_json(metadata_path)


def get_checkpoint_info(checkpoint_path: Path) -> dict:
    """Get comprehensive checkpoint information.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint info including metadata if available
    """
    info: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "step": None,
        "size_bytes": 0,
        "metadata": None,
    }
    
    if checkpoint_path.exists():
        info["size_bytes"] = get_checkpoint_size(checkpoint_path)
        try:
            info["step"] = get_checkpoint_step(checkpoint_path)
        except ValueError:
            pass
        
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata:
            info["metadata"] = metadata
    
    return info


def validate_checkpoint(checkpoint_path: Path) -> tuple[bool, str | None]:
    """Validate checkpoint file integrity.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not checkpoint_path.exists():
        return False, f"Checkpoint file does not exist: {checkpoint_path}"
    
    if not checkpoint_path.is_file():
        return False, f"Checkpoint path is not a file: {checkpoint_path}"
    
    size = get_checkpoint_size(checkpoint_path)
    if size == 0:
        return False, f"Checkpoint file is empty: {checkpoint_path}"
    
    if size < 100:  # Very small files are likely corrupted
        return False, f"Checkpoint file is suspiciously small ({size} bytes): {checkpoint_path}"
    
    return True, None


def validate_checkpoint_state(
    state: TrainState,
    state_template: TrainState,
) -> tuple[bool, str | None]:
    """Validate checkpoint state structure matches template.
    
    Args:
        state: Loaded state to validate
        state_template: Template state for comparison
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check that state has required attributes
    if not hasattr(state, "params"):
        return False, "State missing 'params' attribute"
    
    if not hasattr(state, "opt_state"):
        return False, "State missing 'opt_state' attribute"
    
    # Check parameter structure matches
    def check_structure(path: tuple, node1: Any, node2: Any) -> tuple[bool, str | None]:
        if isinstance(node1, (jnp.ndarray, np.ndarray)) and isinstance(
            node2, (jnp.ndarray, np.ndarray)
        ):
            if node1.shape != node2.shape:
                param_path = "/".join(str(p) for p in path)
                return False, f"Shape mismatch at {param_path}: {node1.shape} != {node2.shape}"
            if node1.dtype != node2.dtype:
                param_path = "/".join(str(p) for p in path)
                return False, f"Dtype mismatch at {param_path}: {node1.dtype} != {node2.dtype}"
        return True, None
    
    # Compare parameter structures
    params1 = state.params
    params2 = state_template.params
    
    def validate_tree(path: tuple, node1: Any, node2: Any) -> tuple[bool, str | None]:
        if isinstance(node1, dict) and isinstance(node2, dict):
            keys1 = set(node1.keys())
            keys2 = set(node2.keys())
            if keys1 != keys2:
                param_path = "/".join(str(p) for p in path) if path else "root"
                missing = keys2 - keys1
                extra = keys1 - keys2
                msg_parts = []
                if missing:
                    msg_parts.append(f"missing keys: {missing}")
                if extra:
                    msg_parts.append(f"extra keys: {extra}")
                return False, f"Key mismatch at {param_path}: {', '.join(msg_parts)}"
            
            for key in keys1:
                result = validate_tree(path + (key,), node1[key], node2[key])
                if not result[0]:
                    return result
        else:
            result = check_structure(path, node1, node2)
            if not result[0]:
                return result
        
        return True, None
    
    return validate_tree((), params1, params2)


def validate_checkpoint_compatibility(
    checkpoint_path: Path,
    config: TrainFlowConfig,
) -> tuple[bool, list[str]]:
    """Validate checkpoint compatibility with current config.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Current training configuration
        
    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    warnings: list[str] = []
    metadata = load_checkpoint_metadata(checkpoint_path)
    
    if not metadata:
        # No metadata available, can't validate
        warnings.append("No metadata available for compatibility check")
        return True, warnings
    
    # Check critical fields from metadata if available
    model_info = metadata.get("model_info", {})
    param_shapes = model_info.get("param_shapes", {})
    
    # Critical fields that must match
    critical_fields = {
        "noise_dimension": config.noise_dimension,
        "condition_dimension": config.condition_dimension,
        "latent_dimension": config.latent_dimension,
        "num_blocks": config.num_blocks,
    }
    
    # Check config hash if available
    if metadata.get("config_hash"):
        config_dict = config.to_dict()
        current_hash = compute_config_hash(config_dict)
        if metadata["config_hash"] != current_hash:
            warnings.append("Config hash mismatch - config may have changed")
    
    # Fields that can differ but should warn
    warn_fields = {
        "batch_size": config.batch_size,
        "base_lr": config.base_lr,
    }
    
    # Note: We can't fully validate without loading the checkpoint,
    # but we can check metadata if it contains config info
    # For now, we'll rely on the config hash check
    
    return True, warnings


def save_checkpoint_with_metadata(
    path: Path,
    state: TrainState | flax_train_state.TrainState,
    step: int,
    config: TrainFlowConfig | None = None,
) -> None:
    """Save checkpoint with metadata.
    
    Args:
        path: Path to save checkpoint
        state: Training state
        step: Training step number
        config: Training configuration (optional, for metadata)
    """
    # Save checkpoint
    save_checkpoint(path, state)
    
    # Save metadata
    save_checkpoint_metadata(path, step, state, config)


def make_run_dir(config: TrainFlowConfig) -> Path:
    """Create a run directory path from config.

    Note: This function is kept for backward compatibility.
    Configs now set workdir in __post_init__, so this just returns
    config.workdir.
    """
    return config.workdir


def plot_samples(
    samples: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    vmin: float = -1.0,
    vmax: float = 1.0,
    dpi: int = 150,
) -> None:
    """Plot a grid of samples with labels."""
    n_show = len(samples)
    n_rows = int(np.ceil(np.sqrt(n_show)))
    n_cols = int(np.ceil(n_show / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 6))
    axs = axs.flatten() if n_show > 1 else [axs]

    for ax, sample, label in zip(axs[:n_show], samples, labels):
        ax.imshow(sample, vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(int(label))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_metrics_row(path: Path, row: dict) -> None:
    """Write metrics row to CSV (append mode, writes header if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_flow_state(
    config: dict, checkpoint_path: Path, batch_size: int
) -> tuple[ConditionalFlow, TrainState]:
    """Load a flow model and state from checkpoint."""
    model = ConditionalFlow(
        noise_dimension=config["noise_dimension"],
        condition_dimension=config["condition_dimension"],
        latent_dimension=config["latent_dimension"],
        num_blocks=config["num_blocks"],
    )
    dummy_x = jnp.zeros(
        (batch_size, config["noise_dimension"]), dtype=jnp.float32
    )
    dummy_t = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    # Initialize both encoder and decoder paths
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    # Initialize encoder
    params_enc = model.init(key1, dummy_x, method="encode")["params"]
    # Initialize decoder (with dummy latents)
    dummy_latents = jnp.zeros(
        (batch_size, config["latent_dimension"]), dtype=jnp.float32
    )
    params_dec = model.init(key2, dummy_x, dummy_t, dummy_latents)["params"]
    # Merge: use encoder params from encode init
    params = {**params_dec, "encoder": params_enc["encoder"]}
    lr = float(config.get("base_lr", 1e-4))
    wd = float(config.get("weight_decay", 1e-6))
    tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = load_checkpoint(checkpoint_path, state)
    return model, state


class LogWriter:
    """Write structured JSON logs in JSONL format.

    One JSON object per line.
    """

    def __init__(self, log_path: Path):
        """Initialize log writer with path to JSONL file."""
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = log_path.open("a", encoding="utf-8")

    def write_step(self, step: int, metrics: dict) -> None:
        """Write a log entry for a training step.

        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
        """
        log_entry = {"step": step, **metrics}
        json_line = json.dumps(log_entry, sort_keys=True)
        self.file.write(json_line + "\n")
        self.file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def find_latest_checkpoint(workdir: Path) -> Path | None:
    """Find the latest checkpoint in workdir/checkpoints/ directory."""
    checkpoints_dir = workdir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    checkpoint_files = list(checkpoints_dir.glob("step_*.msgpack"))
    if not checkpoint_files:
        return None

    # Extract step numbers and find the latest
    latest_checkpoint = None
    latest_step = -1

    for checkpoint_path in checkpoint_files:
        step = get_checkpoint_step(checkpoint_path)
        if step > latest_step:
            latest_step = step
            latest_checkpoint = checkpoint_path

    return latest_checkpoint


def get_checkpoint_step(checkpoint_path: Path) -> int:
    """Extract step number from checkpoint filename.

    Example: step_00100.msgpack -> 100
    """
    match = re.search(r"step_(\d+)\.msgpack", checkpoint_path.name)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Could not extract step number from: {checkpoint_path}"
    )


def unwrap_checkpoint(state: TrainState) -> dict:
    """Extract model parameters from TrainState for inference.
    
    Args:
        state: Training state
        
    Returns:
        Dictionary with only model parameters (no optimizer state)
    """
    return {"params": state.params}


def save_unwrapped_checkpoint(
    path: Path,
    params: dict,
) -> None:
    """Save lightweight checkpoint with only model parameters.
    
    Args:
        path: Path to save unwrapped checkpoint
        params: Model parameters dictionary
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(serialization.to_bytes(params))


def load_unwrapped_checkpoint(path: Path) -> dict:
    """Load unwrapped checkpoint (parameters only).
    
    Args:
        path: Path to unwrapped checkpoint file
        
    Returns:
        Dictionary with model parameters
    """
    with path.open("rb") as f:
        data = f.read()
    return serialization.from_bytes({}, data)


def find_valid_checkpoint(
    workdir: Path,
    state_template: TrainState | None = None,
) -> Path | None:
    """Find latest valid checkpoint, skipping corrupted ones.
    
    Args:
        workdir: Working directory
        state_template: Optional state template for validation
        
    Returns:
        Path to valid checkpoint, or None if none found
    """
    checkpoints_dir = workdir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    checkpoint_files = sorted(
        checkpoints_dir.glob("step_*.msgpack"),
        key=lambda p: get_checkpoint_step(p),
        reverse=True,
    )
    
    for checkpoint_path in checkpoint_files:
        is_valid, error = validate_checkpoint(checkpoint_path)
        if not is_valid:
            continue
        
        if state_template is not None:
            try:
                state = load_checkpoint(checkpoint_path, state_template)
                is_valid, error = validate_checkpoint_state(state, state_template)
                if not is_valid:
                    continue
            except Exception as e:
                # Checkpoint failed to load
                continue
        
        return checkpoint_path
    
    return None


def load_checkpoint_and_resume(
    workdir: Path,
    state_template: TrainState,
    config: TrainFlowConfig | None = None,
) -> tuple[TrainState, int]:
    """Load checkpoint from workdir and return state with starting step.
    
    Enhanced version with validation and error recovery.
    
    Args:
        workdir: Working directory
        state_template: Template state for loading
        config: Optional config for compatibility checking
        
    Returns:
        Tuple of (state, starting_step)
        
    Raises:
        FileNotFoundError: If no valid checkpoint found
        ValueError: If checkpoint validation fails
    """
    checkpoint_path = find_valid_checkpoint(workdir, state_template)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No valid checkpoint found in {workdir / 'checkpoints'}"
        )
    
    # Validate checkpoint file
    is_valid, error = validate_checkpoint(checkpoint_path)
    if not is_valid:
        raise ValueError(f"Checkpoint validation failed: {error}")
    
    # Load checkpoint
    try:
        state = load_checkpoint(checkpoint_path, state_template)
    except Exception as e:
        raise ValueError(
            f"Failed to load checkpoint {checkpoint_path}: {e}"
        ) from e
    
    # Validate state structure
    is_valid, error = validate_checkpoint_state(state, state_template)
    if not is_valid:
        raise ValueError(f"Checkpoint state validation failed: {error}")
    
    # Check compatibility if config provided
    if config:
        is_compatible, warnings = validate_checkpoint_compatibility(checkpoint_path, config)
        if warnings:
            print("Checkpoint compatibility warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    
    starting_step = get_checkpoint_step(checkpoint_path)
    return state, starting_step


def list_checkpoints(workdir: Path) -> list[dict]:
    """List all checkpoints with metadata.
    
    Args:
        workdir: Working directory
        
    Returns:
        List of checkpoint info dictionaries, sorted by step
    """
    checkpoints_dir = workdir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    
    checkpoint_files = list(checkpoints_dir.glob("step_*.msgpack"))
    checkpoints = []
    
    for checkpoint_path in checkpoint_files:
        try:
            info = get_checkpoint_info(checkpoint_path)
            checkpoints.append(info)
        except Exception:
            # Skip invalid checkpoints
            continue
    
    # Sort by step
    checkpoints.sort(key=lambda x: x.get("step", -1) if x.get("step") is not None else -1)
    return checkpoints


def cleanup_old_checkpoints(
    workdir: Path,
    max_checkpoints_to_keep: int,
    keep_final: bool = True,
    final_step: int | None = None,
) -> list[Path]:
    """Remove old checkpoints, keeping only N latest.
    
    Args:
        workdir: Working directory
        max_checkpoints_to_keep: Maximum number of checkpoints to keep
        keep_final: If True, always keep final checkpoint (step = final_step)
        final_step: Step number of final checkpoint (if keep_final is True)
        
    Returns:
        List of removed checkpoint paths
    """
    checkpoints_dir = workdir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    
    checkpoint_files = list(checkpoints_dir.glob("step_*.msgpack"))
    if len(checkpoint_files) <= max_checkpoints_to_keep:
        return []
    
    # Get checkpoint info and sort by step
    checkpoint_info = []
    for checkpoint_path in checkpoint_files:
        try:
            step = get_checkpoint_step(checkpoint_path)
            checkpoint_info.append((step, checkpoint_path))
        except ValueError:
            continue
    
    checkpoint_info.sort(key=lambda x: x[0])
    
    # Identify checkpoints to keep
    to_keep: set[Path] = set()
    
    # Always keep final checkpoint if specified
    if keep_final and final_step is not None:
        for step, path in checkpoint_info:
            if step == final_step:
                to_keep.add(path)
                break
    
    # Keep latest N checkpoints
    for step, path in checkpoint_info[-max_checkpoints_to_keep:]:
        to_keep.add(path)
    
    # Remove old checkpoints
    removed: list[Path] = []
    for step, checkpoint_path in checkpoint_info:
        if checkpoint_path not in to_keep:
            try:
                checkpoint_path.unlink()
                removed.append(checkpoint_path)
                
                # Also remove metadata file if it exists
                metadata_path = get_checkpoint_metadata_path(checkpoint_path)
                if metadata_path.exists():
                    metadata_path.unlink()
            except Exception:
                pass
    
    return removed


@dataclass
class ExperimentMetadata:
    """Experiment metadata for tracking and reproducibility."""

    timestamp: str
    git_commit: str | None = None
    git_commit_short: str | None = None
    config_hash: str | None = None
    python_version: str | None = None
    jax_version: str | None = None
    platform: str | None = None
    processor_name: str | None = None
    cpu_count: int | None = None
    device_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "git_commit_short": self.git_commit_short,
            "config_hash": self.config_hash,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "platform": self.platform,
            "processor_name": self.processor_name,
            "cpu_count": self.cpu_count,
            "device_info": self.device_info,
        }


def get_git_commit_hash() -> tuple[str | None, str | None]:
    """Get git commit hash and short hash.

    Returns:
        Tuple of (full_hash, short_hash) or (None, None) if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path.cwd(),
        )
        full_hash = result.stdout.strip()
        if not full_hash:
            return None, None
        short_hash = full_hash[:7] if len(full_hash) >= 7 else full_hash
        return full_hash, short_hash
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None


def compute_config_hash(config: dict) -> str:
    """Compute SHA256 hash of config for uniqueness.

    Args:
        config: Configuration dictionary

    Returns:
        Hex digest of config hash
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def collect_experiment_metadata(config: TrainFlowConfig) -> ExperimentMetadata:
    """Collect experiment metadata for tracking.

    Args:
        config: Training configuration

    Returns:
        ExperimentMetadata with collected information
    """
    timestamp = datetime.now().isoformat()
    git_commit, git_commit_short = get_git_commit_hash()
    config_dict = config.to_dict()
    config_hash = compute_config_hash(config_dict)

    # Environment info
    python_version = sys.version.split()[0]
    try:
        jax_version = jax.__version__
    except (ImportError, AttributeError):
        jax_version = None

    # System info
    platform_info = f"{platform.system()} {platform.release()}"
    
    # Processor name
    try:
        processor_name = platform.processor()
    except (AttributeError, OSError):
        processor_name = None
    
    # CPU count
    try:
        cpu_count = os.cpu_count() or multiprocessing.cpu_count()
    except (AttributeError, OSError):
        cpu_count = None

    # Device info
    device_info: dict[str, Any] = {}
    try:
        devices = jax.devices()
        device_info["device_count"] = len(devices)
        device_info["devices"] = [str(d) for d in devices]
        if devices:
            device_info["default_device"] = str(devices[0])
            device_info["default_backend"] = devices[0].platform
    except (AttributeError, RuntimeError, OSError):
        pass

    return ExperimentMetadata(
        timestamp=timestamp,
        git_commit=git_commit,
        git_commit_short=git_commit_short,
        config_hash=config_hash,
        python_version=python_version,
        jax_version=jax_version,
        platform=platform_info,
        processor_name=processor_name,
        cpu_count=cpu_count,
        device_info=device_info,
    )


class MetricsAggregator:
    """Aggregate metrics from JSONL training logs."""

    def __init__(self, log_path: Path):
        """Initialize aggregator with path to JSONL log file.

        Args:
            log_path: Path to train_log.jsonl file
        """
        self.log_path = log_path
        self.metrics: list[dict] = []

    def load_metrics(self) -> None:
        """Load all metrics from JSONL file."""
        if not self.log_path.exists():
            return

        self.metrics = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.metrics.append(entry)
                except json.JSONDecodeError:
                    continue

    def get_best_metric(self, metric_name: str, minimize: bool = True) -> dict | None:
        """Get entry with best (min or max) value for a metric.

        Args:
            metric_name: Name of metric to find best value for
            minimize: If True, find minimum; if False, find maximum

        Returns:
            Dictionary entry with best metric value, or None if not found
        """
        if not self.metrics:
            return None

        best_entry = None
        best_value = None

        for entry in self.metrics:
            if metric_name not in entry:
                continue
            value = entry[metric_name]
            if not isinstance(value, (int, float)):
                continue

            if best_value is None:
                best_value = value
                best_entry = entry
            elif minimize and value < best_value:
                best_value = value
                best_entry = entry
            elif not minimize and value > best_value:
                best_value = value
                best_entry = entry

        return best_entry

    def get_final_metrics(self) -> dict:
        """Get final metrics from last log entry.

        Returns:
            Dictionary of final metrics
        """
        if not self.metrics:
            return {}
        return self.metrics[-1].copy()

    def get_metric_trend(self, metric_name: str) -> list[float]:
        """Get trend of a metric over training.

        Args:
            metric_name: Name of metric to extract

        Returns:
            List of metric values in order
        """
        values = []
        for entry in self.metrics:
            if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                values.append(float(entry[metric_name]))
        return values

    def get_metric_statistics(self, metric_name: str) -> dict[str, float] | None:
        """Get statistics for a metric (mean, std, min, max, percentiles).

        Args:
            metric_name: Name of metric to analyze

        Returns:
            Dictionary with statistics, or None if metric not found
        """
        values = self.get_metric_trend(metric_name)
        if not values:
            return None

        values_array = np.array(values)
        stats: dict[str, float] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
        }

        # Percentiles
        for p in [10, 25, 75, 90]:
            stats[f"p{p}"] = float(np.percentile(values_array, p))

        return stats


def generate_training_summary(log_path: Path) -> dict:
    """Generate training summary from JSONL logs.

    Args:
        log_path: Path to train_log.jsonl file

    Returns:
        Dictionary with training summary statistics
    """
    aggregator = MetricsAggregator(log_path)
    aggregator.load_metrics()

    if not aggregator.metrics:
        return {"error": "No metrics found in log file"}

    summary: dict[str, Any] = {}

    # Best metrics
    best_loss = aggregator.get_best_metric("loss", minimize=True)
    if best_loss:
        summary["best_loss"] = {
            "value": best_loss["loss"],
            "step": best_loss.get("step", -1),
        }

    best_loss_avg = aggregator.get_best_metric("loss_avg", minimize=True)
    if best_loss_avg:
        summary["best_loss_avg"] = {
            "value": best_loss_avg["loss_avg"],
            "step": best_loss_avg.get("step", -1),
        }

    # Final metrics
    final_metrics = aggregator.get_final_metrics()
    summary["final_metrics"] = final_metrics

    # Training statistics
    if aggregator.metrics:
        steps = [m.get("step", -1) for m in aggregator.metrics if "step" in m]
        if steps:
            summary["total_steps"] = max(steps)
            summary["logged_steps"] = len(steps)

    # Convergence tracking (check if loss is decreasing)
    loss_trend = aggregator.get_metric_trend("loss")
    if len(loss_trend) > 10:
        early_avg = sum(loss_trend[:10]) / 10
        late_avg = sum(loss_trend[-10:]) / 10
        summary["convergence"] = {
            "early_avg_loss": early_avg,
            "late_avg_loss": late_avg,
            "improvement": early_avg - late_avg,
            "improvement_percent": ((early_avg - late_avg) / early_avg * 100) if early_avg > 0 else 0.0,
        }

    # Loss statistics
    loss_stats = aggregator.get_metric_statistics("loss")
    if loss_stats:
        summary["loss_statistics"] = loss_stats

    return summary


def compare_configs(config1: dict, config2: dict) -> dict:
    """Compare two config dictionaries and identify differences.

    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary

    Returns:
        Dictionary with differences: added, removed, changed keys, with type info
    """
    keys1 = set(config1.keys())
    keys2 = set(config2.keys())

    added = keys2 - keys1
    removed = keys1 - keys2
    common = keys1 & keys2

    changed: dict[str, dict[str, Any]] = {}
    for key in common:
        val1 = config1[key]
        val2 = config2[key]
        if val1 != val2:
            change_info: dict[str, Any] = {
                "old": val1,
                "new": val2,
                "old_type": type(val1).__name__,
                "new_type": type(val2).__name__,
            }
            
            # Add magnitude for numeric changes
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                change_info["absolute_change"] = abs(val2 - val1)
                if val1 != 0:
                    change_info["relative_change"] = abs((val2 - val1) / val1) * 100
            
            changed[key] = change_info

    return {
        "added": list(added),
        "removed": list(removed),
        "changed": changed,
        "unchanged": list(common - set(changed.keys())),
    }


def generate_config_diff(workdir: Path, new_config: TrainFlowConfig) -> dict | None:
    """Generate config diff between previous run and new config.

    Args:
        workdir: Work directory (may contain previous config.json)
        new_config: New configuration to compare

    Returns:
        Dictionary with config diff, or None if no previous config found
    """
    prev_config_path = workdir / "config.json"
    if not prev_config_path.exists():
        return None

    try:
        prev_config = load_json(prev_config_path)
        new_config_dict = new_config.to_dict()
        diff = compare_configs(prev_config, new_config_dict)
        return diff
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
        # Log but don't fail - return None to indicate diff unavailable
        return None


def generate_experiment_tag(config: TrainFlowConfig) -> str:
    """Generate experiment tag/name from config.

    Args:
        config: Training configuration

    Returns:
        Experiment tag string
    """
    parts = []

    # Method
    if config.method:
        parts.append(config.method)
    elif config.use_improved_mean_flow:
        parts.append("improved_mean_flow")
    else:
        parts.append("mean_flow")

    # Architecture
    if config.architecture:
        parts.append(config.architecture)
    else:
        parts.append("unknown")

    # Dataset
    if config.dataset:
        parts.append(config.dataset)
    else:
        parts.append("unknown")

    # Timestamp (short format)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)

    # Git commit (short)
    _, git_commit_short = get_git_commit_hash()
    if git_commit_short:
        parts.append(git_commit_short)

    return "--".join(parts)


class ProgressBar:
    """Simple progress bar for training progress.

    Falls back to simple prints if tqdm is not available.
    """

    def __init__(self, total: int, desc: str = "Training", disable: bool = False):
        """Initialize progress bar.

        Args:
            total: Total number of steps
            desc: Description string
            disable: If True, disable progress bar
        """
        self.total = total
        self.desc = desc
        self.disable = disable
        self.current = 0
        self._tqdm = None

        if not disable:
            try:
                from tqdm import tqdm
                self._tqdm = tqdm(total=total, desc=desc, unit="step")
            except ImportError:
                self._tqdm = None

    def update(self, n: int = 1) -> None:
        """Update progress by n steps.

        Args:
            n: Number of steps to advance
        """
        self.current += n
        if self._tqdm:
            self._tqdm.update(n)
        elif not self.disable and self.current % max(1, self.total // 20) == 0:
            percent = (self.current / self.total) * 100
            print(f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)")

    def set_postfix(self, **kwargs) -> None:
        """Set postfix information (e.g., loss values).

        Args:
            **kwargs: Key-value pairs to display
        """
        if self._tqdm:
            self._tqdm.set_postfix(**kwargs)

    def close(self) -> None:
        """Close progress bar."""
        if self._tqdm:
            self._tqdm.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def plot_loss_curve(
    log_path: Path,
    output_path: Path,
    metric_name: str = "loss",
    title: str | None = None,
) -> None:
    """Plot loss curve from JSONL logs.

    Args:
        log_path: Path to train_log.jsonl file
        output_path: Path to save plot
        metric_name: Name of metric to plot
        title: Plot title (defaults to metric_name)
    """
    aggregator = MetricsAggregator(log_path)
    aggregator.load_metrics()

    if not aggregator.metrics:
        return

    steps = []
    values = []

    for entry in aggregator.metrics:
        if "step" in entry and metric_name in entry:
            step = entry["step"]
            value = entry[metric_name]
            if isinstance(value, (int, float)):
                steps.append(step)
                values.append(float(value))

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, values, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"{metric_name.replace('_', ' ').title()} over Training")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
