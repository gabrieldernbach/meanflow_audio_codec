"""Tests for hierarchical configuration system."""

from pathlib import Path

import pytest

from meanflow_audio_codec.configs.config import (
    AudioConfig,
    BaseConfig,
    DatasetConfig,
    MethodConfig,
    MNISTConfig,
    ModelConfig,
    TrainFlowConfig,
    TrainingConfig,
    create_audio_config,
    create_mnist_config,
    diff_configs,
    load_config_from_json,
    merge_configs,
    migrate_config_v1_to_v2,
)


def test_base_config_validation():
    """Test BaseConfig validation."""
    # Valid config
    config = BaseConfig(
        batch_size=128,
        n_steps=10000,
        base_lr=0.0001,
        weight_decay=0.0001,
        seed=42,
    )
    config.validate()
    
    # Invalid batch_size
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        config = BaseConfig(
            batch_size=0,
            n_steps=10000,
            base_lr=0.0001,
            weight_decay=0.0001,
            seed=42,
        )
        config.validate()


def test_model_config_validation():
    """Test ModelConfig validation."""
    # Valid config
    config = ModelConfig(
        noise_dimension=784,
        condition_dimension=128,
        latent_dimension=256,
        num_blocks=8,
    )
    config.validate()
    
    # Invalid condition_dimension (odd)
    with pytest.raises(ValueError, match="condition_dimension must be even"):
        config = ModelConfig(
            noise_dimension=784,
            condition_dimension=127,
            latent_dimension=256,
            num_blocks=8,
        )
        config.validate()


def test_method_config_validation():
    """Test MethodConfig validation."""
    # Valid config
    config = MethodConfig(
        use_improved_mean_flow=False,
        method="flow_matching",
    )
    config.validate()
    
    # Invalid method
    with pytest.raises(ValueError, match="method must be one of"):
        config = MethodConfig(
            use_improved_mean_flow=False,
            method="invalid_method",
        )
        config.validate()
    
    # Cross-field validation: improved_mean_flow requires use_improved_mean_flow=True
    with pytest.raises(ValueError, match="requires use_improved_mean_flow=True"):
        config = MethodConfig(
            use_improved_mean_flow=False,
            method="improved_mean_flow",
        )
        config.validate()


def test_train_flow_config_creation():
    """Test TrainFlowConfig creation."""
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
    dataset = DatasetConfig(dataset="mnist")
    method = MethodConfig(use_improved_mean_flow=False)
    training = TrainingConfig(
        sample_every=1000,
        sample_seed=42,
        sample_steps=50,
        workdir=Path("./outputs/test"),
    )
    
    config = TrainFlowConfig(
        base=base,
        model=model,
        dataset=dataset,
        method=method,
        training=training,
    )
    
    assert config.batch_size == 128
    assert config.workdir == Path("./outputs/test")
    assert config.config_version == "2.0"


def test_backward_compatibility_properties():
    """Test backward compatibility properties."""
    config = create_mnist_config()
    
    # Test flat access works
    assert config.batch_size == 128
    assert config.n_steps == 10000
    assert config.base_lr == 0.0001
    assert config.noise_dimension == 784
    assert config.condition_dimension == 128
    assert config.use_improved_mean_flow is False


def test_config_migration():
    """Test config migration from v1.0 to v2.0."""
    v1_config = {
        "batch_size": 128,
        "n_steps": 10000,
        "sample_every": 1000,
        "sample_seed": 42,
        "sample_steps": 50,
        "base_lr": 0.0001,
        "weight_decay": 0.0001,
        "seed": 42,
        "use_improved_mean_flow": True,
        "workdir": "./outputs/test",
        "checkpoint_step": None,
        "data_dir": None,
        "noise_dimension": 784,
        "condition_dimension": 128,
        "latent_dimension": 256,
        "num_blocks": 8,
    }
    
    v2_config = migrate_config_v1_to_v2(v1_config)
    
    assert v2_config["config_version"] == "2.0"
    assert "base" in v2_config
    assert "model" in v2_config
    assert "dataset" in v2_config
    assert "method" in v2_config
    assert "training" in v2_config
    assert v2_config["base"]["batch_size"] == 128
    assert v2_config["method"]["use_improved_mean_flow"] is True


def test_config_loading():
    """Test loading config from JSON file."""
    # Create a temporary v1.0 config file
    import tempfile
    import json
    
    v1_config = {
        "batch_size": 128,
        "n_steps": 10000,
        "sample_every": 1000,
        "sample_seed": 42,
        "sample_steps": 50,
        "base_lr": 0.0001,
        "weight_decay": 0.0001,
        "seed": 42,
        "use_improved_mean_flow": False,
        "workdir": "./outputs/test",
        "checkpoint_step": None,
        "data_dir": None,
        "noise_dimension": 784,
        "condition_dimension": 128,
        "latent_dimension": 256,
        "num_blocks": 8,
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(v1_config, f)
        temp_path = Path(f.name)
    
    try:
        config = load_config_from_json(temp_path)
        assert config.config_version == "2.0"
        assert config.batch_size == 128
        assert config.workdir == Path("./outputs/test")
    finally:
        temp_path.unlink()


def test_config_diff():
    """Test config diff utility."""
    config1 = create_mnist_config()
    config2 = create_mnist_config()
    config2.base.batch_size = 256
    
    diff = diff_configs(config1, config2)
    
    assert "changed" in diff
    assert len(diff["changed"]) > 0
    assert "base.batch_size" in diff["changed"]


def test_config_merging():
    """Test config merging."""
    base_config = create_mnist_config()
    override = {"base": {"batch_size": 256}}
    
    merged = merge_configs(base_config, override)
    
    assert merged.batch_size == 256
    assert merged.n_steps == 10000  # Unchanged


def test_factory_functions():
    """Test factory functions."""
    mnist_config = create_mnist_config()
    assert mnist_config.dataset == "mnist"
    assert mnist_config.noise_dimension == 784
    
    audio_config = create_audio_config()
    assert audio_config.dataset == "audio"
    assert audio_config.noise_dimension == 256 * 256 * 3


def test_schema_generation():
    """Test schema generation."""
    config = create_mnist_config()
    schema = config.get_schema()
    
    assert "config_version" in schema
    assert "base" in schema
    assert "model" in schema
    assert "dataset" in schema
    assert "method" in schema
    assert "training" in schema


def test_documentation_generation():
    """Test documentation generation."""
    config = create_mnist_config()
    docs = config.get_documentation()
    
    assert "# TrainFlowConfig Documentation" in docs
    assert "BaseConfig" in docs
    assert "ModelConfig" in docs


