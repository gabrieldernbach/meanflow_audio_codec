"""Noise schedule strategies for flow matching.

Noise schedules define how to interpolate between data and noise,
and how to compute velocity targets.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""
    
    @abstractmethod
    def interpolate(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate between x0 and x1 at time t.
        
        Args:
            x0: Data samples [B, ...]
            x1: Noise samples [B, ...]
            t: Time values [B, 1] or [B]
        
        Returns:
            Interpolated samples [B, ...]
        """
        pass
    
    @abstractmethod
    def compute_target(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute velocity target from x0 and x1.
        
        Args:
            x0: Data samples [B, ...]
            x1: Noise samples [B, ...]
        
        Returns:
            Target velocity [B, ...]
        """
        pass


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule with configurable min/max noise levels.
    
    Interpolates as: (1-t) * x0 + (noise_min + noise_max * t) * x1
    Target: noise_max * x1 - x0
    """
    
    def __init__(self, noise_min: float = 0.001, noise_max: float = 0.999):
        """Initialize linear noise schedule.
        
        Args:
            noise_min: Minimum noise level (at t=0)
            noise_max: Maximum noise level (at t=1)
        """
        self.noise_min = noise_min
        self.noise_max = noise_max
    
    def interpolate(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linear interpolation with noise scaling."""
        # Ensure t is [B, 1]
        if t.ndim == 1:
            t = t[:, None]
        noise_scale = self.noise_min + self.noise_max * t
        return (1.0 - t) * x0 + noise_scale * x1
    
    def compute_target(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute velocity target: noise_max * x1 - x0."""
        return self.noise_max * x1 - x0


class UniformNoiseSchedule(NoiseSchedule):
    """Uniform noise schedule (standard flow matching).
    
    Interpolates as: (1-t) * x0 + t * x1
    Target: x1 - x0
    """
    
    def interpolate(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Uniform interpolation."""
        if t.ndim == 1:
            t = t[:, None]
        return (1.0 - t) * x0 + t * x1
    
    def compute_target(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute velocity target: x1 - x0."""
        return x1 - x0

