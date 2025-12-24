from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    noise_dim: int = 28 * 28
    cond_dim: int = 512
    latent_dim: int = 1024
    n_blocks: int = 10
    n_classes: int = 10


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 512
    steps: int = 8_000
    warmup: int = 100
    device: str = 'mps'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    sample_n_steps: int = 100
    ema_beta: float = 0.99
    ema_alpha: float = 0.01
    log_frequency: int = 50
    eval_frequency: int = 200
    eval_steps: int = 100
    figsize: tuple[int, int] = (6, 6)


@dataclass
class FlowStrategyConfig:
    """Flow strategy specific configuration."""
    # Flow matching specific
    noise_min: float = 0.001
    noise_max: float = 0.999
    
    # Mean Flow / Improved Mean Flow specific
    flow_ratio: float = 0.5
    gamma: float = 0.5
    c: float = 1e-3
