import json
import warnings
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


@dataclass
class TrainFlowConfig:
    batch_size: int
    n_steps: int
    sample_every: int
    sample_seed: int
    sample_steps: int
    base_lr: float
    weight_decay: float
    seed: int
    use_improved_mean_flow: bool
    checkpoint_step: int | None
    data_dir: str | None
    noise_dimension: int
    condition_dimension: int
    latent_dimension: int
    num_blocks: int
    workdir: Path | None = None
    # Tokenization
    tokenization_strategy: str | None = None  # "mdct" or "reshape"
    tokenization_config: dict | None = None  # Strategy-specific parameters
    # Method, architecture, dataset selection
    method: str | None = None  # "autoencoder", "flow_matching", "mean_flow", "improved_mean_flow"
    architecture: str | None = None  # "mlp", "mlp_mixer", "convnet"
    dataset: str | None = None  # "mnist", "audio"
    # Improved Mean Flow hyperparameters
    gamma: float | None = None  # Default 1.0
    flow_ratio: float | None = None  # Default 1.0
    c: float | None = None  # Default 1.0
    # Ablation flags
    use_stop_gradient: bool | None = None  # Default True
    loss_weighting: str | None = None  # "uniform", "time_dependent", "learned"
    # Composability options
    loss_strategy: str | None = None  # "flow_matching", "mean_flow", "improved_mean_flow" (default: inferred from use_improved_mean_flow)
    noise_schedule: str | None = None  # "linear", "uniform" (default: "linear")
    noise_min: float | None = None  # Default: 0.001
    noise_max: float | None = None  # Default: 0.999
    time_sampling: str | None = None  # "uniform", "logit_normal", "mean_flow" (default: "logit_normal")
    time_sampling_mean: float | None = None  # Default: -0.4
    time_sampling_std: float | None = None  # Default: 1.0
    time_sampling_data_proportion: float | None = None  # Default: 0.5 (for mean_flow)
    use_weighted_loss: bool | None = None  # Default: True
    # Deprecated fields for backward compatibility
    output_dir: Path | None = None
    run_name: str | None = None

    def __post_init__(self):
        """Handle backward compatibility and workdir setup."""
        # Handle deprecated output_dir/run_name -> workdir migration
        if self.workdir is None:
            if self.output_dir is not None:
                warnings.warn(
                    "output_dir and run_name are deprecated. "
                    "Use workdir instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if self.run_name is not None:
                    run_name = self.run_name
                else:
                    tag = (
                        "improved"
                        if self.use_improved_mean_flow
                        else "baseline"
                    )
                    run_name = f"seed{self.seed}_{tag}"
                self.workdir = self.output_dir / run_name
            else:
                raise ValueError(
                    "Either workdir or output_dir must be provided"
                )

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # Skip deprecated fields and None values
            if field.name in ("output_dir", "run_name"):
                continue
            if value is None:
                continue
            # Convert Path to string
            if isinstance(value, Path):
                result[field.name] = str(value)
            else:
                result[field.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TrainFlowConfig":
        """Create config from dictionary (e.g., loaded from JSON)."""
        # Convert string paths to Path objects
        if "workdir" in data and data["workdir"] is not None:
            data["workdir"] = Path(data["workdir"])
        if "output_dir" in data and data["output_dir"] is not None:
            data["output_dir"] = Path(data["output_dir"])
        # Filter out None values for optional fields only
        # Required fields that can be None (checkpoint_step, data_dir) must be preserved
        required_fields = {f.name for f in fields(cls) if f.default == MISSING}
        filtered = {
            k: v
            for k, v in data.items()
            if v is not None or k in required_fields
        }
        return cls(**filtered)


@dataclass
class EvaluationConfig:
    checkpoint: Path
    config_path: Path | None
    output_dir: Path | None
    n_steps: list[int]
    num_samples: int
    batch_size: int
    seed: int
    metrics_csv: Path
    data_dir: str | None
    real_split: str
    use_improved_mean_flow: bool | None
    noise_dimension: int | None
    condition_dimension: int | None
    latent_dimension: int | None
    num_blocks: int | None


@dataclass
class AnalysisConfig:
    metrics_csv: Path
    workdir: Path | None = None
    # Deprecated field for backward compatibility
    output_dir: Path | None = None

    def __post_init__(self):
        """Handle backward compatibility."""
        if self.workdir is None:
            if self.output_dir is not None:
                warnings.warn(
                    "output_dir is deprecated. Use workdir instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.workdir = self.output_dir
            else:
                raise ValueError(
                    "Either workdir or output_dir must be provided"
                )

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "output_dir":
                continue
            if value is None:
                continue
            if isinstance(value, Path):
                result[field.name] = str(value)
            else:
                result[field.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """Create config from dictionary."""
        if "workdir" in data:
            data["workdir"] = Path(data["workdir"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        filtered = {k: v for k, v in data.items() if v is not None}
        return cls(**filtered)


def load_config_from_json(path: Path) -> TrainFlowConfig:
    """Load TrainFlowConfig from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainFlowConfig.from_dict(data)
