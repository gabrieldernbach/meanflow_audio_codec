"""Hierarchical configuration system with validation, schema, and migration support."""

import json
import warnings
from dataclasses import MISSING, dataclass, fields, field
from pathlib import Path
from typing import Any


# ============================================================================
# Base Config Classes
# ============================================================================


@dataclass
class BaseConfig:
    """Base configuration with core training parameters."""
    
    batch_size: int
    n_steps: int
    base_lr: float
    weight_decay: float
    seed: int
    
    def validate(self) -> None:
        """Validate base configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def get_schema(self) -> dict:
        """Get schema metadata for this config."""
        schema = {}
        for f in fields(self):
            schema[f.name] = {
                "type": str(f.type),
                "required": f.default == MISSING,
                "default": f.default if f.default != MISSING else None,
            }
        return schema


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    noise_dimension: int
    condition_dimension: int
    latent_dimension: int
    num_blocks: int
    architecture: str | None = None
    
    def validate(self) -> None:
        """Validate model configuration parameters."""
        if self.noise_dimension <= 0:
            raise ValueError(f"noise_dimension must be > 0, got {self.noise_dimension}")
        if self.condition_dimension <= 0:
            raise ValueError(f"condition_dimension must be > 0, got {self.condition_dimension}")
        if self.condition_dimension % 2 != 0:
            raise ValueError(f"condition_dimension must be even, got {self.condition_dimension}")
        if self.latent_dimension <= 0:
            raise ValueError(f"latent_dimension must be > 0, got {self.latent_dimension}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {self.num_blocks}")
        if self.architecture is not None:
            valid_architectures = ["mlp", "mlp_mixer", "convnet"]
            if self.architecture not in valid_architectures:
                raise ValueError(
                    f"architecture must be one of {valid_architectures}, got {self.architecture}"
                )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result
    
    def get_schema(self) -> dict:
        """Get schema metadata for this config."""
        schema = {}
        for f in fields(self):
            schema[f.name] = {
                "type": str(f.type),
                "required": f.default == MISSING,
                "default": f.default if f.default != MISSING else None,
            }
        if self.architecture is not None:
            schema["architecture"]["allowed_values"] = ["mlp", "mlp_mixer", "convnet"]
        return schema


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    dataset: str | None = None
    data_dir: str | None = None
    tokenization_strategy: str | None = None
    tokenization_config: dict | None = None
    
    def validate(self) -> None:
        """Validate dataset configuration parameters."""
        if self.dataset is not None:
            valid_datasets = ["mnist", "audio"]
            if self.dataset not in valid_datasets:
                raise ValueError(
                    f"dataset must be one of {valid_datasets}, got {self.dataset}"
                )
        if self.tokenization_strategy is not None:
            valid_strategies = ["mdct", "reshape"]
            if self.tokenization_strategy not in valid_strategies:
                raise ValueError(
                    f"tokenization_strategy must be one of {valid_strategies}, "
                    f"got {self.tokenization_strategy}"
                )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result
    
    def get_schema(self) -> dict:
        """Get schema metadata for this config."""
        schema = {}
        for f in fields(self):
            schema[f.name] = {
                "type": str(f.type),
                "required": f.default == MISSING,
                "default": f.default if f.default != MISSING else None,
            }
        schema["dataset"]["allowed_values"] = ["mnist", "audio"]
        schema["tokenization_strategy"]["allowed_values"] = ["mdct", "reshape"]
        return schema


@dataclass
class MethodConfig:
    """Method-specific configuration."""
    
    method: str | None = None
    use_improved_mean_flow: bool = False
    gamma: float | None = None
    flow_ratio: float | None = None
    c: float | None = None
    use_stop_gradient: bool | None = None
    loss_weighting: str | None = None
    loss_strategy: str | None = None
    noise_schedule: str | None = None
    noise_min: float | None = None
    noise_max: float | None = None
    time_sampling: str | None = None
    time_sampling_mean: float | None = None
    time_sampling_std: float | None = None
    time_sampling_data_proportion: float | None = None
    use_weighted_loss: bool | None = None
    
    def validate(self) -> None:
        """Validate method configuration parameters."""
        if self.method is not None:
            valid_methods = ["autoencoder", "flow_matching", "mean_flow", "improved_mean_flow"]
            if self.method not in valid_methods:
                raise ValueError(
                    f"method must be one of {valid_methods}, got {self.method}"
                )
        if self.loss_strategy is not None:
            valid_strategies = ["flow_matching", "mean_flow", "improved_mean_flow"]
            if self.loss_strategy not in valid_strategies:
                raise ValueError(
                    f"loss_strategy must be one of {valid_strategies}, got {self.loss_strategy}"
                )
        if self.noise_schedule is not None:
            valid_schedules = ["linear", "uniform"]
            if self.noise_schedule not in valid_schedules:
                raise ValueError(
                    f"noise_schedule must be one of {valid_schedules}, got {self.noise_schedule}"
                )
        if self.time_sampling is not None:
            valid_sampling = ["uniform", "logit_normal", "mean_flow"]
            if self.time_sampling not in valid_sampling:
                raise ValueError(
                    f"time_sampling must be one of {valid_sampling}, got {self.time_sampling}"
                )
        if self.loss_weighting is not None:
            valid_weighting = ["uniform", "time_dependent", "learned"]
            if self.loss_weighting not in valid_weighting:
                raise ValueError(
                    f"loss_weighting must be one of {valid_weighting}, got {self.loss_weighting}"
                )
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.flow_ratio is not None and self.flow_ratio <= 0:
            raise ValueError(f"flow_ratio must be > 0, got {self.flow_ratio}")
        if self.c is not None and self.c <= 0:
            raise ValueError(f"c must be > 0, got {self.c}")
        if self.noise_min is not None and (self.noise_min < 0 or self.noise_min >= 1):
            raise ValueError(f"noise_min must be in [0, 1), got {self.noise_min}")
        if self.noise_max is not None and (self.noise_max <= 0 or self.noise_max > 1):
            raise ValueError(f"noise_max must be in (0, 1], got {self.noise_max}")
        if self.noise_min is not None and self.noise_max is not None:
            if self.noise_min >= self.noise_max:
                raise ValueError(
                    f"noise_min ({self.noise_min}) must be < noise_max ({self.noise_max})"
                )
        if self.time_sampling_std is not None and self.time_sampling_std <= 0:
            raise ValueError(f"time_sampling_std must be > 0, got {self.time_sampling_std}")
        if self.time_sampling_data_proportion is not None:
            if not 0 <= self.time_sampling_data_proportion <= 1:
                raise ValueError(
                    f"time_sampling_data_proportion must be in [0, 1], "
                    f"got {self.time_sampling_data_proportion}"
                )
        # Cross-field validation
        if self.method == "improved_mean_flow" and not self.use_improved_mean_flow:
            raise ValueError(
                "method='improved_mean_flow' requires use_improved_mean_flow=True"
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result
    
    def get_schema(self) -> dict:
        """Get schema metadata for this config."""
        schema = {}
        for f in fields(self):
            schema[f.name] = {
                "type": str(f.type),
                "required": f.default == MISSING,
                "default": f.default if f.default != MISSING else None,
            }
        schema["method"]["allowed_values"] = ["autoencoder", "flow_matching", "mean_flow", "improved_mean_flow"]
        schema["loss_strategy"]["allowed_values"] = ["flow_matching", "mean_flow", "improved_mean_flow"]
        schema["noise_schedule"]["allowed_values"] = ["linear", "uniform"]
        schema["time_sampling"]["allowed_values"] = ["uniform", "logit_normal", "mean_flow"]
        schema["loss_weighting"]["allowed_values"] = ["uniform", "time_dependent", "learned"]
        return schema


@dataclass
class TrainingConfig:
    """Training infrastructure configuration."""
    
    sample_every: int
    sample_seed: int
    sample_steps: int
    workdir: Path | None = None
    checkpoint_step: int | None = None
    max_checkpoints_to_keep: int | None = None
    
    def validate(self) -> None:
        """Validate training configuration parameters."""
        if self.sample_every <= 0:
            raise ValueError(f"sample_every must be > 0, got {self.sample_every}")
        if self.sample_steps <= 0:
            raise ValueError(f"sample_steps must be > 0, got {self.sample_steps}")
        if self.checkpoint_step is not None and self.checkpoint_step <= 0:
            raise ValueError(f"checkpoint_step must be > 0, got {self.checkpoint_step}")
        if self.max_checkpoints_to_keep is not None and self.max_checkpoints_to_keep <= 0:
            raise ValueError(
                f"max_checkpoints_to_keep must be > 0, got {self.max_checkpoints_to_keep}"
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, Path):
                result[f.name] = str(value)
            else:
                result[f.name] = value
        return result
    
    def get_schema(self) -> dict:
        """Get schema metadata for this config."""
        schema = {}
        for f in fields(self):
            schema[f.name] = {
                "type": str(f.type),
                "required": f.default == MISSING,
                "default": f.default if f.default != MISSING else None,
            }
        return schema


# ============================================================================
# Dataset-Specific Configs
# ============================================================================


@dataclass
class MNISTConfig(DatasetConfig):
    """MNIST-specific dataset configuration."""
    
    dataset: str = field(default="mnist", init=False)
    noise_dimension: int = field(default=784, init=False)  # 28*28
    
    def __post_init__(self):
        """Set MNIST-specific defaults."""
        super().__init__()
        object.__setattr__(self, "dataset", "mnist")
        if self.tokenization_strategy is None:
            object.__setattr__(self, "tokenization_strategy", "reshape")


@dataclass
class AudioConfig(DatasetConfig):
    """Audio-specific dataset configuration."""
    
    dataset: str = field(default="audio", init=False)
    
    def __post_init__(self):
        """Set audio-specific defaults."""
        super().__init__()
        object.__setattr__(self, "dataset", "audio")
        if self.tokenization_strategy is None:
            object.__setattr__(self, "tokenization_strategy", "mdct")


# ============================================================================
# Main Config Class
# ============================================================================


@dataclass(init=False)
class TrainFlowConfig:
    """Complete training configuration with hierarchical structure."""
    
    base: BaseConfig
    model: ModelConfig
    dataset: DatasetConfig
    method: MethodConfig
    training: TrainingConfig
    # Deprecated fields for backward compatibility  
    output_dir: Path | None = None
    run_name: str | None = None
    config_version: str = "2.0"
    
    def __init__(
        self,
        base: BaseConfig,
        model: ModelConfig,
        dataset: DatasetConfig,
        method: MethodConfig,
        training: TrainingConfig,
        output_dir: Path | None = None,
        run_name: str | None = None,
        config_version: str = "2.0",
    ):
        """Initialize TrainFlowConfig."""
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_training", training)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "config_version", config_version)
        self.__post_init__()
    
    def __post_init__(self):
        """Handle backward compatibility and validation."""
        # Handle deprecated output_dir/run_name -> workdir migration
        dataset_config = object.__getattribute__(self, "_dataset")
        if self.training.workdir is None:
            if self.output_dir is not None:
                warnings.warn(
                    "output_dir and run_name are deprecated. Use workdir instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if self.run_name is not None:
                    run_name = self.run_name
                else:
                    tag = (
                        "improved"
                        if self.method.use_improved_mean_flow
                        else "baseline"
                    )
                    run_name = f"seed{self.base.seed}_{tag}"
                object.__setattr__(self.training, "workdir", self.output_dir / run_name)
            else:
                raise ValueError("Either workdir or output_dir must be provided")
        
        # Validate all sub-configs
        self.validate()
    
    def validate(self) -> None:
        """Validate complete configuration."""
        self.base.validate()
        self.model.validate()
        dataset_config = object.__getattribute__(self, "_dataset")
        dataset_config.validate()
        method_config = object.__getattribute__(self, "_method")
        method_config.validate()
        self.training.validate()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        dataset_config = object.__getattribute__(self, "_dataset")
        method_config = object.__getattribute__(self, "_method")
        result = {
            "config_version": self.config_version,
            "base": self.base.to_dict(),
            "model": self.model.to_dict(),
            "dataset": dataset_config.to_dict(),
            "method": method_config.to_dict(),
            "training": self.training.to_dict(),
        }
        # Skip deprecated fields
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainFlowConfig":
        """Create config from dictionary (e.g., loaded from JSON)."""
        # Detect config version
        config_version = data.get("config_version", "1.0")
        
        # Check if it's flat format (v1.0) by looking for top-level fields
        is_flat = "base" not in data and any(k in data for k in ["batch_size", "n_steps", "base_lr"])
        
        if is_flat or config_version == "1.0":
            # Migrate from flat v1.0 format
            data = migrate_config_v1_to_v2(data)
        
        # Handle hierarchical structure
        if "base" in data:
            # New hierarchical format
            base = BaseConfig(**data["base"])
            model = ModelConfig(**data["model"])
            dataset = DatasetConfig(**data["dataset"])
            method = MethodConfig(**data["method"])
            training_data = data["training"].copy()
            if "workdir" in training_data and training_data["workdir"] is not None:
                training_data["workdir"] = Path(training_data["workdir"])
            training = TrainingConfig(**training_data)
        else:
            # Should have been migrated, but handle gracefully
            raise ValueError("Invalid config format: expected hierarchical structure")
        
        # Handle deprecated fields
        output_dir = None
        run_name = None
        if "output_dir" in data and data["output_dir"] is not None:
            output_dir = Path(data["output_dir"])
        if "run_name" in data:
            run_name = data["run_name"]
        
        result = cls(
            base=base,
            model=model,
            dataset=dataset,
            method=method,
            training=training,
            output_dir=output_dir,
            run_name=run_name,
        )
        # Set config_version after creation
        object.__setattr__(result, "config_version", data.get("config_version", "2.0"))
        return result
    
    def get_schema(self) -> dict:
        """Get complete schema for this config."""
        dataset_config = object.__getattribute__(self, "_dataset")
        method_config = object.__getattribute__(self, "_method")
        return {
            "config_version": self.config_version,
            "base": self.base.get_schema(),
            "model": self.model.get_schema(),
            "dataset": dataset_config.get_schema(),
            "method": method_config.get_schema(),
            "training": self.training.get_schema(),
        }
    
    def get_documentation(self) -> str:
        """Get human-readable documentation for this config."""
        lines = ["# TrainFlowConfig Documentation", ""]
        lines.append(f"Config Version: {self.config_version}")
        lines.append("")
        
        dataset_config = object.__getattribute__(self, "_dataset")
        method_config = object.__getattribute__(self, "_method")
        for section_name, section_config in [
            ("Base", self.base),
            ("Model", self.model),
            ("Dataset", dataset_config),
            ("Method", method_config),
            ("Training", self.training),
        ]:
            lines.append(f"## {section_name}Config")
            lines.append("")
            schema = section_config.get_schema()
            for field_name, field_info in schema.items():
                lines.append(f"- `{field_name}`: {field_info['type']}")
                if field_info.get("allowed_values"):
                    lines.append(f"  - Allowed values: {', '.join(field_info['allowed_values'])}")
                if field_info["default"] is not None:
                    lines.append(f"  - Default: {field_info['default']}")
                lines.append("")
        
        return "\n".join(lines)
    
    # Backward compatibility: provide properties for flat access
    @property
    def base(self) -> BaseConfig:
        return object.__getattribute__(self, "_base")
    
    @property
    def model(self) -> ModelConfig:
        return object.__getattribute__(self, "_model")
    
    @property
    def method(self) -> MethodConfig:
        return object.__getattribute__(self, "_method")
    
    @property
    def training(self) -> TrainingConfig:
        return object.__getattribute__(self, "_training")
    
    @property
    def batch_size(self) -> int:
        return self.base.batch_size
    
    @property
    def n_steps(self) -> int:
        return self.base.n_steps
    
    @property
    def base_lr(self) -> float:
        return self.base.base_lr
    
    @property
    def weight_decay(self) -> float:
        return self.base.weight_decay
    
    @property
    def seed(self) -> int:
        return self.base.seed
    
    @property
    def noise_dimension(self) -> int:
        return self.model.noise_dimension
    
    @property
    def condition_dimension(self) -> int:
        return self.model.condition_dimension
    
    @property
    def latent_dimension(self) -> int:
        return self.model.latent_dimension
    
    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks
    
    @property
    def architecture(self) -> str | None:
        return self.model.architecture
    
    @property
    def dataset(self) -> str | None:
        """Dataset name (backward compatibility property)."""
        # Access underlying attribute to avoid recursion
        dataset_config = object.__getattribute__(self, "_dataset")
        return dataset_config.dataset if dataset_config else None
    
    @property
    def data_dir(self) -> str | None:
        dataset_config = object.__getattribute__(self, "_dataset")
        return dataset_config.data_dir
    
    @property
    def tokenization_strategy(self) -> str | None:
        dataset_config = object.__getattribute__(self, "_dataset")
        return dataset_config.tokenization_strategy
    
    @property
    def tokenization_config(self) -> dict | None:
        dataset_config = object.__getattribute__(self, "_dataset")
        return dataset_config.tokenization_config
    
    @property
    def method(self) -> str | None:
        """Method name (backward compatibility property)."""
        method_config = object.__getattribute__(self, "_method")
        return method_config.method if method_config else None
    
    @property
    def use_improved_mean_flow(self) -> bool:
        method_config = object.__getattribute__(self, "_method")
        return method_config.use_improved_mean_flow
    
    @property
    def gamma(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.gamma
    
    @property
    def flow_ratio(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.flow_ratio
    
    @property
    def c(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.c
    
    @property
    def use_stop_gradient(self) -> bool | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.use_stop_gradient
    
    @property
    def loss_weighting(self) -> str | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.loss_weighting
    
    @property
    def loss_strategy(self) -> str | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.loss_strategy
    
    @property
    def noise_schedule(self) -> str | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.noise_schedule
    
    @property
    def noise_min(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.noise_min
    
    @property
    def noise_max(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.noise_max
    
    @property
    def time_sampling(self) -> str | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.time_sampling
    
    @property
    def time_sampling_mean(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.time_sampling_mean
    
    @property
    def time_sampling_std(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.time_sampling_std
    
    @property
    def time_sampling_data_proportion(self) -> float | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.time_sampling_data_proportion
    
    @property
    def use_weighted_loss(self) -> bool | None:
        method_config = object.__getattribute__(self, "_method")
        return method_config.use_weighted_loss
    
    @property
    def workdir(self) -> Path | None:
        return self.training.workdir
    
    @property
    def checkpoint_step(self) -> int | None:
        return self.training.checkpoint_step
    
    @property
    def sample_every(self) -> int:
        return self.training.sample_every
    
    @property
    def sample_seed(self) -> int:
        return self.training.sample_seed
    
    @property
    def sample_steps(self) -> int:
        return self.training.sample_steps
    
    @property
    def max_checkpoints_to_keep(self) -> int | None:
        return self.training.max_checkpoints_to_keep


# ============================================================================
# Migration Utilities
# ============================================================================


def migrate_config_v1_to_v2(data: dict) -> dict:
    """Migrate flat v1.0 config to hierarchical v2.0 format.
    
    Args:
        data: Flat config dictionary (v1.0 format)
    
    Returns:
        Hierarchical config dictionary (v2.0 format)
    """
    # Extract base config
    base = {
        "batch_size": data["batch_size"],
        "n_steps": data["n_steps"],
        "base_lr": data["base_lr"],
        "weight_decay": data["weight_decay"],
        "seed": data["seed"],
    }
    
    # Extract model config
    model = {
        "noise_dimension": data["noise_dimension"],
        "condition_dimension": data["condition_dimension"],
        "latent_dimension": data["latent_dimension"],
        "num_blocks": data["num_blocks"],
    }
    if "architecture" in data:
        model["architecture"] = data["architecture"]
    
    # Extract dataset config
    dataset = {}
    if "dataset" in data:
        dataset["dataset"] = data["dataset"]
    if "data_dir" in data:
        dataset["data_dir"] = data["data_dir"]
    if "tokenization_strategy" in data:
        dataset["tokenization_strategy"] = data["tokenization_strategy"]
    if "tokenization_config" in data:
        dataset["tokenization_config"] = data["tokenization_config"]
    
    # Extract method config
    method = {
        "use_improved_mean_flow": data.get("use_improved_mean_flow", False),
    }
    if "method" in data:
        method["method"] = data["method"]
    if "gamma" in data:
        method["gamma"] = data["gamma"]
    if "flow_ratio" in data:
        method["flow_ratio"] = data["flow_ratio"]
    if "c" in data:
        method["c"] = data["c"]
    if "use_stop_gradient" in data:
        method["use_stop_gradient"] = data["use_stop_gradient"]
    if "loss_weighting" in data:
        method["loss_weighting"] = data["loss_weighting"]
    if "loss_strategy" in data:
        method["loss_strategy"] = data["loss_strategy"]
    if "noise_schedule" in data:
        method["noise_schedule"] = data["noise_schedule"]
    if "noise_min" in data:
        method["noise_min"] = data["noise_min"]
    if "noise_max" in data:
        method["noise_max"] = data["noise_max"]
    if "time_sampling" in data:
        method["time_sampling"] = data["time_sampling"]
    if "time_sampling_mean" in data:
        method["time_sampling_mean"] = data["time_sampling_mean"]
    if "time_sampling_std" in data:
        method["time_sampling_std"] = data["time_sampling_std"]
    if "time_sampling_data_proportion" in data:
        method["time_sampling_data_proportion"] = data["time_sampling_data_proportion"]
    if "use_weighted_loss" in data:
        method["use_weighted_loss"] = data["use_weighted_loss"]
    
    # Extract training config
    training = {
        "sample_every": data["sample_every"],
        "sample_seed": data["sample_seed"],
        "sample_steps": data["sample_steps"],
    }
    if "workdir" in data:
        training["workdir"] = data["workdir"]
    if "checkpoint_step" in data:
        training["checkpoint_step"] = data["checkpoint_step"]
    
    # Build hierarchical structure
    result = {
        "config_version": "2.0",
        "base": base,
        "model": model,
        "dataset": dataset,
        "method": method,
        "training": training,
    }
    
    # Preserve deprecated fields
    if "output_dir" in data:
        result["output_dir"] = data["output_dir"]
    if "run_name" in data:
        result["run_name"] = data["run_name"]
    
    return result


# ============================================================================
# Config Merging and Factory Functions
# ============================================================================


def merge_configs(base: TrainFlowConfig, override: dict) -> TrainFlowConfig:
    """Merge override dictionary into base config.
    
    Args:
        base: Base configuration
        override: Dictionary with overrides (can be partial, hierarchical or flat)
    
    Returns:
        New merged configuration
    """
    # Convert base to dict
    base_dict = base.to_dict()
    
    # Handle flat override format (for backward compatibility)
    if "base" not in override and any(k in override for k in ["batch_size", "n_steps", "base_lr"]):
        # Flat format - migrate first
        override = migrate_config_v1_to_v2(override)
    
    # Deep merge
    def deep_merge(base_dict: dict, override_dict: dict) -> dict:
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = deep_merge(base_dict, override)
    return TrainFlowConfig.from_dict(merged)


def create_mnist_config(**overrides) -> TrainFlowConfig:
    """Create a default MNIST configuration.
    
    Args:
        **overrides: Override any default values (can be flat or hierarchical)
    
    Returns:
        TrainFlowConfig for MNIST
    """
    base = BaseConfig(
        batch_size=128,
        n_steps=10000,
        base_lr=0.0001,
        weight_decay=0.0001,
        seed=42,
    )
    model = ModelConfig(
        noise_dimension=784,
        condition_dimension=128,
        latent_dimension=256,
        num_blocks=8,
    )
    dataset = DatasetConfig(
        dataset="mnist",
        tokenization_strategy="reshape",
    )
    method = MethodConfig(
        use_improved_mean_flow=False,
    )
    training = TrainingConfig(
        sample_every=1000,
        sample_seed=42,
        sample_steps=50,
        workdir=Path("./outputs/audio_default"),
    )
    
    config = TrainFlowConfig(
        base=base,
        model=model,
        dataset=dataset,
        method=method,
        training=training,
    )
    
    # Apply overrides if provided
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


def create_audio_config(**overrides) -> TrainFlowConfig:
    """Create a default audio configuration.
    
    Args:
        **overrides: Override any default values (can be flat or hierarchical)
    
    Returns:
        TrainFlowConfig for audio
    """
    base = BaseConfig(
        batch_size=128,
        n_steps=10000,
        base_lr=0.0001,
        weight_decay=0.0001,
        seed=42,
    )
    model = ModelConfig(
        noise_dimension=256 * 256 * 3,  # frame_sz * n_channels
        condition_dimension=128,
        latent_dimension=256,
        num_blocks=8,
    )
    dataset = DatasetConfig(
        dataset="audio",
        tokenization_strategy="mdct",
    )
    method = MethodConfig(
        use_improved_mean_flow=False,
    )
    training = TrainingConfig(
        sample_every=1000,
        sample_seed=42,
        sample_steps=50,
        workdir=Path("./outputs/mnist_default"),
    )
    
    config = TrainFlowConfig(
        base=base,
        model=model,
        dataset=dataset,
        method=method,
        training=training,
    )
    
    # Apply overrides if provided
    if overrides:
        config = merge_configs(config, overrides)
    
    return config


# ============================================================================
# Config Diff Utilities
# ============================================================================


def diff_configs(config1: TrainFlowConfig, config2: TrainFlowConfig) -> dict:
    """Compare two configs and return differences.
    
    Args:
        config1: First configuration
        config2: Second configuration
    
    Returns:
        Dictionary with 'changed', 'added', and 'removed' keys
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat1 = flatten_dict(dict1)
    flat2 = flatten_dict(dict2)
    
    changed = {}
    for key in set(flat1.keys()) & set(flat2.keys()):
        if flat1[key] != flat2[key]:
            changed[key] = {"old": flat1[key], "new": flat2[key]}
    
    added = [key for key in flat2.keys() if key not in flat1]
    removed = [key for key in flat1.keys() if key not in flat2]
    
    return {
        "changed": changed,
        "added": added,
        "removed": removed,
    }


def print_config_diff(diff: dict) -> None:
    """Print config diff in human-readable format.
    
    Args:
        diff: Diff dictionary from diff_configs()
    """
    if diff["changed"]:
        print("Changed parameters:")
        for key, change in diff["changed"].items():
            print(f"  {key}: {change['old']} -> {change['new']}")
    
    if diff["added"]:
        print(f"Added parameters: {', '.join(diff['added'])}")
    
    if diff["removed"]:
        print(f"Removed parameters: {', '.join(diff['removed'])}")
    
    if not diff["changed"] and not diff["added"] and not diff["removed"]:
        print("No differences found.")


# ============================================================================
# Legacy Config Classes (for backward compatibility)
# ============================================================================


@dataclass
class EvaluationConfig:
    """Evaluation configuration (unchanged for now)."""
    
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
    """Analysis configuration (unchanged for now)."""
    
    metrics_csv: Path
    workdir: Path | None = None
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
                raise ValueError("Either workdir or output_dir must be provided")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "output_dir":
                continue
            if value is None:
                continue
            if isinstance(value, Path):
                result[f.name] = str(value)
            else:
                result[f.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """Create from dictionary."""
        if "workdir" in data:
            data["workdir"] = Path(data["workdir"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        filtered = {k: v for k, v in data.items() if v is not None}
        return cls(**filtered)


# ============================================================================
# Load Functions
# ============================================================================


def load_config_from_json(path: Path) -> TrainFlowConfig:
    """Load TrainFlowConfig from a JSON file.
    
    Supports both v1.0 (flat) and v2.0 (hierarchical) formats.
    Automatically migrates v1.0 configs to v2.0.
    
    Args:
        path: Path to JSON config file
    
    Returns:
        TrainFlowConfig instance
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainFlowConfig.from_dict(data)
