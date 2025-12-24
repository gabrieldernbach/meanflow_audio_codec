"""Shared utilities for training, evaluation, and checkpointing."""

import csv
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import serialization
from flax.training import train_state as flax_train_state

from meanflow_audio_codec.configs.config import TrainFlowConfig
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


def load_checkpoint_and_resume(
    workdir: Path, state_template: TrainState
) -> tuple[TrainState, int]:
    """Load checkpoint from workdir and return state with starting step."""
    checkpoint_path = find_latest_checkpoint(workdir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {workdir / 'checkpoints'}"
        )

    state = load_checkpoint(checkpoint_path, state_template)
    starting_step = get_checkpoint_step(checkpoint_path)
    return state, starting_step
