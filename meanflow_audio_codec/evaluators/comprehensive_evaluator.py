"""Comprehensive evaluator for computing all metrics and tracking performance.

This module provides a unified interface for evaluating trained models,
computing all relevant metrics, and tracking performance characteristics.
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from meanflow_audio_codec.configs.config import TrainFlowConfig, load_config_from_json
from meanflow_audio_codec.evaluators import audio_metrics, metrics
from meanflow_audio_codec.evaluators.performance import (
    count_parameters,
    inference_time,
    memory_usage,
)
from meanflow_audio_codec.evaluators.sampling import sample
from meanflow_audio_codec.trainers.utils import load_checkpoint, load_flow_state


class ComprehensiveEvaluator:
    """Comprehensive evaluator that computes all metrics and tracks performance."""
    
    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        dataset: str = "mnist",
    ):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (if None, tries to load from checkpoint)
            dataset: Dataset type ("mnist" or "audio")
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.dataset = dataset
        
        # Load config
        if config_path is not None:
            self.config = load_config_from_json(config_path)
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = checkpoint_path.parent
            config_file = checkpoint_dir.parent / "config.json"
            if config_file.exists():
                self.config = load_config_from_json(config_file)
            else:
                raise ValueError(
                    "config_path must be provided or config.json must exist in checkpoint directory"
                )
        
        # Load model
        self.model, self.state = load_flow_state(
            self.config.to_dict(),
            checkpoint_path,
            batch_size=self.config.batch_size,
        )
        
        # Count parameters
        self.param_count = count_parameters(self.state.params)
    
    def evaluate(
        self,
        real_data: np.ndarray,
        num_samples: int = 1000,
        n_steps_list: list[int] = [1, 10, 50, 250],
        batch_size: int = 32,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run comprehensive evaluation.
        
        Args:
            real_data: Real data samples, shape [N, ...]
            num_samples: Number of samples to generate
            n_steps_list: List of NFE values to test
            batch_size: Batch size for generation
            seed: Random seed
        
        Returns:
            Dictionary with all metrics and performance data
        """
        results: dict[str, Any] = {
            "config": {
                "method": self.config.method,
                "architecture": self.config.architecture,
                "dataset": self.config.dataset,
                "tokenization": self.config.tokenization_strategy,
            },
            "parameters": self.param_count,
            "memory_before": memory_usage(),
            "nfe_results": {},
        }
        
        key = jax.random.PRNGKey(seed)
        
        # Generate samples for each NFE value
        for n_steps in n_steps_list:
            nfe_results: dict[str, Any] = {}
            
            # Generate samples
            print(f"Generating {num_samples} samples with {n_steps} steps...")
            generated_samples = []
            
            # Create dummy latents (for conditional models)
            # In practice, these would come from an encoder
            dummy_latents = jnp.zeros(
                (batch_size, self.config.latent_dimension), dtype=jnp.float32
            )
            
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                actual_batch_size = batch_end - batch_start
                
                # Adjust latents for last batch
                if actual_batch_size < batch_size:
                    batch_latents = dummy_latents[:actual_batch_size]
                else:
                    batch_latents = dummy_latents
                
                key, sample_key = jax.random.split(key)
                batch_samples = sample(
                    self.state.apply_fn,
                    self.model.noise_dimension,
                    self.state.params,
                    sample_key,
                    latents=batch_latents,
                    n_steps=n_steps,
                    use_improved_mean_flow=self.config.use_improved_mean_flow,
                    guidance_scale=1.0,
                )
                
                generated_samples.append(np.array(batch_samples[:actual_batch_size]))
            
            generated = np.concatenate(generated_samples, axis=0)[:num_samples]
            
            # Measure inference time
            def sample_fn(key):
                return sample(
                    self.state.apply_fn,
                    self.model.noise_dimension,
                    self.state.params,
                    key,
                    latents=dummy_latents[:1],
                    n_steps=n_steps,
                    use_improved_mean_flow=self.config.use_improved_mean_flow,
                    guidance_scale=1.0,
                )
            
            timing_results = inference_time(sample_fn, key, num_warmup=5, num_runs=50)
            nfe_results["inference_time"] = timing_results
            
            # Compute metrics
            # Ensure same number of samples
            n_eval = min(len(real_data), len(generated))
            real_eval = real_data[:n_eval]
            gen_eval = generated[:n_eval]
            
            # MSE
            if real_eval.ndim > 1:
                mse = np.mean((real_eval - gen_eval) ** 2)
            else:
                mse = np.mean((real_eval - gen_eval) ** 2)
            nfe_results["mse"] = float(mse)
            
            # PSNR and SSIM (for images)
            if self.dataset == "mnist":
                # Reshape to images if needed
                if real_eval.ndim == 2 and real_eval.shape[1] == 784:
                    real_images = real_eval.reshape(n_eval, 28, 28)
                    gen_images = gen_eval.reshape(n_eval, 28, 28)
                else:
                    real_images = real_eval
                    gen_images = gen_eval
                
                # PSNR
                psnr_val = metrics.psnr(gen_images, real_images)
                nfe_results["psnr"] = psnr_val
                
                # SSIM
                ssim_val = metrics.ssim(gen_images, real_images)
                nfe_results["ssim"] = ssim_val
            
            # Audio metrics
            elif self.dataset == "audio":
                # PESQ (requires proper sample rate, may fail if not available)
                try:
                    pesq_val = audio_metrics.pesq_score(
                        real_eval, gen_eval, sample_rate=16000
                    )
                    nfe_results["pesq"] = pesq_val
                except (ImportError, ValueError) as e:
                    nfe_results["pesq"] = None
                    nfe_results["pesq_error"] = str(e)
                
                # STOI
                try:
                    stoi_val = audio_metrics.stoi_score(
                        real_eval, gen_eval, sample_rate=16000
                    )
                    nfe_results["stoi"] = stoi_val
                except (ImportError, ValueError) as e:
                    nfe_results["stoi"] = None
                    nfe_results["stoi_error"] = str(e)
                
                # Spectral distance
                try:
                    spectral_dist = audio_metrics.spectral_distance(
                        real_eval, gen_eval, domain="mdct"
                    )
                    nfe_results["spectral_distance"] = spectral_dist
                except Exception as e:
                    nfe_results["spectral_distance"] = None
                    nfe_results["spectral_distance_error"] = str(e)
            
            # FID and KID require feature extraction (e.g., Inception network)
            # For now, we'll skip these as they require additional setup
            # TODO: Add FID/KID computation with feature extractor
            
            results["nfe_results"][str(n_steps)] = nfe_results
        
        results["memory_after"] = memory_usage()
        
        return results
    
    def save_results(self, results: dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary from evaluate()
            output_path: Path to save JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        serializable_results = convert_to_json_serializable(results)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, sort_keys=True)

