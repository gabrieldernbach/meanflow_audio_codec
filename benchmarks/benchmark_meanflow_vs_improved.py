"""
Benchmark comparing original MeanFlow vs Improved MeanFlow on MNIST.

This script trains both models and compares:
- Training stability (loss curves)
- Convergence speed
- Final model quality (MSE on validation set)
- Sample quality metrics
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict

from meanflow_audio_codec.references.mean_flow_mnist import (
    Config, ConditionalFlow, init_training
)
from meanflow_audio_codec.references.improved_mean_flow_mnist import (
    Config as ImprovedConfig,
    ConditionalFlow as ImprovedConditionalFlow,
    init_training as init_training_improved
)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    batch_size: int = 512
    steps: int = 2_000  # Shorter for quick comparison
    device: str = 'mps'
    eval_steps: int = 50
    eval_frequency: int = 100
    log_frequency: int = 50
    seed: int = 42


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IteratorWrapper:
    """Wrapper for iterators that recreates them when exhausted."""
    def __init__(self, split='train', batch_size=512, seed=42):
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.iterator = self._create_iterator()
    
    def _create_iterator(self):
        from meanflow_audio_codec.datasets.mnist import load_mnist
        from pathlib import Path
        data_dir = str(Path.home() / "datasets" / "mnist")
        return load_mnist(
            data_dir=data_dir,
            split=self.split,
            batch_size=self.batch_size,
            format='1d',
            normalize=True,
            seed=self.seed
        )
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = self._create_iterator()
            return next(self.iterator)
    
    def __iter__(self):
        return self


def train_and_track(model, train_iter_wrapper, val_iter_wrapper, opt, cfg, method_name):
    """Train model and track metrics."""
    print(f"\n{'='*60}")
    print(f"Training {method_name}")
    print(f"{'='*60}\n")
    
    metrics = {
        'train_loss': [],
        'train_loss_ema': [],
        'val_loss': [],
        'val_mse': [],
        'step': []
    }
    
    train_loss_ema = None
    val_loss_ema = None
    
    for i in range(cfg.steps):
        # Training step
        img_np, lbl_np = next(train_iter_wrapper)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        if method_name == "MeanFlow":
            loss, mse = model.mean_flow_loss(
                img, lbl, cfg.flow_ratio, cfg.gamma, cfg.c
            )
        else:  # Improved MeanFlow
            loss, mse = model.improved_mean_flow_loss(
                img, lbl, cfg.flow_ratio
            )

        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss_ema = train_loss_ema * 0.999 + loss.item() * 0.001 if train_loss_ema is not None else loss.item()
        
        # Track metrics
        if i % cfg.log_frequency == 0:
            metrics['train_loss'].append(loss.item())
            metrics['train_loss_ema'].append(train_loss_ema)
            metrics['step'].append(i)
        
        # Periodic validation
        should_eval = (i + 1) % cfg.eval_frequency == 0
        if should_eval:
            # Evaluate inline
            model.eval()
            total_loss = 0.0
            total_mse = 0.0
            n_batches = 0
            
            for _ in range(cfg.eval_steps):
                img_np, lbl_np = next(val_iter_wrapper)
                img = torch.from_numpy(img_np).to(cfg.device)
                lbl = torch.from_numpy(lbl_np).to(cfg.device)
                
                if method_name == "MeanFlow":
                    loss, mse = model.mean_flow_loss(
                        img, lbl, cfg.flow_ratio, cfg.gamma, cfg.c
                    )
                else:
                    loss, mse = model.improved_mean_flow_loss(
                        img, lbl, cfg.flow_ratio
                    )
                
                total_loss += loss.item()
                total_mse += mse.item()
                n_batches += 1
            
            model.train()
            val_loss = total_loss / n_batches if n_batches > 0 else 0.0
            val_mse = total_mse / n_batches if n_batches > 0 else 0.0
            
            val_loss_ema = val_loss_ema * 0.999 + val_loss * 0.001 if val_loss_ema is not None else val_loss
            
            metrics['val_loss'].append(val_loss)
            metrics['val_mse'].append(val_mse)
            
            if i % cfg.log_frequency == 0:
                print(f'{i=:05d}  train_loss={loss.item():.6f}  train_loss_ema={train_loss_ema:.6f}  '
                      f'val_loss={val_loss:.6f}  val_mse={val_mse:.6f}')
            else:
                print(f'{i=:05d}  eval: val_loss={val_loss:.6f}  val_mse={val_mse:.6f}')
    
    return metrics, model


@torch.no_grad()
def evaluate_sample_quality(model, val_iter_wrapper, cfg, n_samples=100, method_name="MeanFlow"):
    """Evaluate sample quality metrics."""
    model.eval()
    
    all_mse = []
    all_loss = []
    
    for _ in range(n_samples):
        img_np, lbl_np = next(val_iter_wrapper)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        if method_name == "MeanFlow":
            loss, mse = model.mean_flow_loss(
                img, lbl, cfg.flow_ratio, cfg.gamma, cfg.c
            )
        else:
            loss, mse = model.improved_mean_flow_loss(
                img, lbl, cfg.flow_ratio
            )
        
        all_mse.append(mse.item())
        all_loss.append(loss.item())
    
    model.train()
    
    return {
        'mean_mse': np.mean(all_mse),
        'std_mse': np.std(all_mse),
        'mean_loss': np.mean(all_loss),
        'std_loss': np.std(all_loss),
    }


@torch.no_grad()
def benchmark_inference_speed(model, cfg, n_runs=100, method_name="MeanFlow"):
    """Benchmark inference speed."""
    model.eval()
    
    # Create dummy labels
    cls_idx = torch.randint(0, 10, (cfg.batch_size,), device=cfg.device)
    
    # Warmup
    for _ in range(10):
        _ = model.sample(cls_idx, n_steps=cfg.sample_n_steps)
    
    # Benchmark
    torch.cuda.synchronize() if cfg.device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(n_runs):
        _ = model.sample(cls_idx, n_steps=cfg.sample_n_steps)
    
    torch.cuda.synchronize() if cfg.device == 'cuda' else None
    end_time = time.time()
    
    model.train()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / n_runs
    avg_time_per_sample = avg_time_per_batch / cfg.batch_size
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_per_second': cfg.batch_size / avg_time_per_batch,
    }


def run_benchmark():
    """Run the full benchmark comparing both methods."""
    set_seed(42)
    
    bench_cfg = BenchmarkConfig()
    
    # Initialize configurations
    cfg_base = Config(
        batch_size=bench_cfg.batch_size,
        steps=bench_cfg.steps,
        device=bench_cfg.device,
        eval_steps=bench_cfg.eval_steps,
        eval_frequency=bench_cfg.eval_frequency,
        log_frequency=bench_cfg.log_frequency,
    )
    
    cfg_improved = ImprovedConfig(
        batch_size=bench_cfg.batch_size,
        steps=bench_cfg.steps,
        device=bench_cfg.device,
        eval_steps=bench_cfg.eval_steps,
        eval_frequency=bench_cfg.eval_frequency,
        log_frequency=bench_cfg.log_frequency,
    )
    
    print("="*60)
    print("MeanFlow vs Improved MeanFlow Benchmark")
    print("="*60)
    
    results = {}
    
    # Train MeanFlow (original)
    set_seed(42)
    train_iter_base, val_iter_base, model_base, opt_base = init_training(cfg_base)
    train_wrapper_base = IteratorWrapper('train', cfg_base.batch_size, 42)
    val_wrapper_base = IteratorWrapper('test', cfg_base.batch_size, 43)
    metrics_base, model_base = train_and_track(
        model_base, train_wrapper_base, val_wrapper_base, opt_base, cfg_base, "MeanFlow"
    )
    results['MeanFlow'] = {
        'metrics': metrics_base,
        'final_val_loss': metrics_base['val_loss'][-1] if metrics_base['val_loss'] else None,
        'final_val_mse': metrics_base['val_mse'][-1] if metrics_base['val_mse'] else None,
    }
    
    # Train Improved MeanFlow
    set_seed(42)
    train_iter_improved, val_iter_improved, model_improved, opt_improved = init_training_improved(cfg_improved)
    train_wrapper_improved = IteratorWrapper('train', cfg_improved.batch_size, 42)
    val_wrapper_improved = IteratorWrapper('test', cfg_improved.batch_size, 43)
    metrics_improved, model_improved = train_and_track(
        model_improved, train_wrapper_improved, val_wrapper_improved, opt_improved, cfg_improved, "Improved MeanFlow"
    )
    results['Improved MeanFlow'] = {
        'metrics': metrics_improved,
        'final_val_loss': metrics_improved['val_loss'][-1] if metrics_improved['val_loss'] else None,
        'final_val_mse': metrics_improved['val_mse'][-1] if metrics_improved['val_mse'] else None,
    }
    
    # Evaluate sample quality
    print("\n" + "="*60)
    print("Evaluating Sample Quality")
    print("="*60)
    
    val_wrapper_base_eval = IteratorWrapper('test', cfg_base.batch_size, 43)
    val_wrapper_improved_eval = IteratorWrapper('test', cfg_improved.batch_size, 43)
    quality_base = evaluate_sample_quality(model_base, val_wrapper_base_eval, cfg_base, method_name="MeanFlow")
    quality_improved = evaluate_sample_quality(model_improved, val_wrapper_improved_eval, cfg_improved, method_name="Improved MeanFlow")
    
    results['MeanFlow']['sample_quality'] = quality_base
    results['Improved MeanFlow']['sample_quality'] = quality_improved
    
    print(f"\nMeanFlow - MSE: {quality_base['mean_mse']:.6f} ± {quality_base['std_mse']:.6f}")
    print(f"Improved MeanFlow - MSE: {quality_improved['mean_mse']:.6f} ± {quality_improved['std_mse']:.6f}")
    
    # Benchmark inference speed
    print("\n" + "="*60)
    print("Benchmarking Inference Speed")
    print("="*60)
    
    speed_base = benchmark_inference_speed(model_base, cfg_base, method_name="MeanFlow")
    speed_improved = benchmark_inference_speed(model_improved, cfg_improved, method_name="Improved MeanFlow")
    
    results['MeanFlow']['inference_speed'] = speed_base
    results['Improved MeanFlow']['inference_speed'] = speed_improved
    
    print(f"\nMeanFlow:")
    print(f"  Avg time per batch: {speed_base['avg_time_per_batch']*1000:.2f} ms")
    print(f"  Samples per second: {speed_base['samples_per_second']:.2f}")
    
    print(f"\nImproved MeanFlow:")
    print(f"  Avg time per batch: {speed_improved['avg_time_per_batch']*1000:.2f} ms")
    print(f"  Samples per second: {speed_improved['samples_per_second']:.2f}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    print(f"\nFinal Validation Loss:")
    print(f"  MeanFlow: {results['MeanFlow']['final_val_loss']:.6f}")
    print(f"  Improved MeanFlow: {results['Improved MeanFlow']['final_val_loss']:.6f}")
    improvement_loss = ((results['MeanFlow']['final_val_loss'] - results['Improved MeanFlow']['final_val_loss']) 
                       / results['MeanFlow']['final_val_loss'] * 100)
    print(f"  Improvement: {improvement_loss:+.2f}%")
    
    print(f"\nFinal Validation MSE:")
    print(f"  MeanFlow: {results['MeanFlow']['final_val_mse']:.6f}")
    print(f"  Improved MeanFlow: {results['Improved MeanFlow']['final_val_mse']:.6f}")
    improvement_mse = ((results['MeanFlow']['final_val_mse'] - results['Improved MeanFlow']['final_val_mse']) 
                      / results['MeanFlow']['final_val_mse'] * 100)
    print(f"  Improvement: {improvement_mse:+.2f}%")
    
    print(f"\nSample Quality (MSE):")
    print(f"  MeanFlow: {quality_base['mean_mse']:.6f} ± {quality_base['std_mse']:.6f}")
    print(f"  Improved MeanFlow: {quality_improved['mean_mse']:.6f} ± {quality_improved['std_mse']:.6f}")
    improvement_quality = ((quality_base['mean_mse'] - quality_improved['mean_mse']) 
                          / quality_base['mean_mse'] * 100)
    print(f"  Improvement: {improvement_quality:+.2f}%")
    
    # Save results
    output_path = Path("benchmarks") / "meanflow_comparison_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    results_serializable = {}
    for method, data in results.items():
        results_serializable[method] = {
            'final_val_loss': data['final_val_loss'],
            'final_val_mse': data['final_val_mse'],
            'sample_quality': data['sample_quality'],
            'inference_speed': data['inference_speed'],
            'metrics': {
                k: v for k, v in data['metrics'].items()
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_benchmark()

