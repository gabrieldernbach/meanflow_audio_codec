"""Profiling utilities for training.

This module provides utilities for tracking training time, memory usage,
and other performance metrics during training.
"""

import time
from typing import Any

from meanflow_audio_codec.evaluators.performance import (
    TrainingTimer,
    count_parameters,
    memory_usage,
)


class ProfilingTrainer:
    """Wrapper for training loop that adds profiling capabilities."""
    
    def __init__(self, log_writer: Any):
        """Initialize profiling trainer.
        
        Args:
            log_writer: LogWriter instance for logging metrics
        """
        self.log_writer = log_writer
        self.step_times: list[float] = []
        self.start_time: float | None = None
        self.last_step_time: float | None = None
    
    def start_training(self, params: Any) -> None:
        """Called at the start of training.
        
        Args:
            params: Model parameters (for counting)
        """
        self.start_time = time.time()
        self.last_step_time = self.start_time
        
        # Count parameters and log once
        param_count = count_parameters(params)
        self.log_writer.write_step(
            0,
            {
                "parameters_total": param_count["total"],
                "parameters_millions": param_count["total_millions"],
                "memory_initial": memory_usage(),
            },
        )
    
    def before_step(self, step: int) -> None:
        """Called before each training step.
        
        Args:
            step: Current step number
        """
        self.last_step_time = time.time()
    
    def after_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Called after each training step.
        
        Args:
            step: Current step number
            metrics: Dictionary of metrics to log (will be enhanced with timing)
        """
        current_time = time.time()
        step_time = current_time - self.last_step_time if self.last_step_time else 0.0
        self.step_times.append(step_time)
        
        # Compute average step time (over last 100 steps)
        avg_step_time = (
            sum(self.step_times[-100:]) / len(self.step_times[-100:])
            if self.step_times
            else 0.0
        )
        
        # Enhanced metrics
        enhanced_metrics = {
            **metrics,
            "step_time": step_time,
            "avg_step_time": avg_step_time,
        }
        
        # Add memory usage periodically (every 100 steps to avoid overhead)
        if step % 100 == 0:
            enhanced_metrics["memory"] = memory_usage()
        
        self.log_writer.write_step(step, enhanced_metrics)
        self.last_step_time = current_time
    
    def end_training(self, step: int) -> dict[str, Any]:
        """Called at the end of training.
        
        Args:
            step: Final step number
        
        Returns:
            Dictionary with training summary statistics
        """
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0.0
        
        summary = {
            "total_training_time": total_time,
            "total_training_time_hours": total_time / 3600.0,
            "avg_step_time": (
                sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
            ),
            "steps_per_second": (
                len(self.step_times) / total_time if total_time > 0 else 0.0
            ),
            "memory_final": memory_usage(),
        }
        
        # Log summary
        self.log_writer.write_step(step, summary)
        
        return summary

