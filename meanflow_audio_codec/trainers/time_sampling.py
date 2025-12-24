"""Time sampling strategies for flow matching.

Time sampling strategies define how to sample time values t (and optionally r)
for training flow models.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp

from meanflow_audio_codec.utils import logit_normal, sample_tr


class TimeSamplingStrategy(ABC):
    """Abstract base class for time sampling strategies."""
    
    @abstractmethod
    def sample_time(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Sample time values.
        
        Args:
            key: Random key
            batch_size: Batch size
            dtype: Output dtype
        
        Returns:
            Time values [B, 1]
        """
        pass


class UniformTimeSampling(TimeSamplingStrategy):
    """Uniform time sampling: t ~ Uniform[0, 1]."""
    
    def sample_time(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Sample time uniformly from [0, 1]."""
        return jax.random.uniform(key, (batch_size, 1), dtype=dtype)


class LogitNormalTimeSampling(TimeSamplingStrategy):
    """Logit-normal time sampling.
    
    Samples from logit-normal distribution, which concentrates
    probability mass near 0 and 1.
    """
    
    def __init__(self, mean: float = -0.4, std: float = 1.0):
        """Initialize logit-normal time sampling.
        
        Args:
            mean: Mean of underlying normal distribution
            std: Standard deviation of underlying normal distribution
        """
        self.mean = mean
        self.std = std
    
    def sample_time(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Sample time from logit-normal distribution."""
        return logit_normal(key, (batch_size, 1), mean=self.mean, std=self.std, dtype=dtype)


class MeanFlowTimeSampling(TimeSamplingStrategy):
    """Time sampling for mean flow methods.
    
    Samples both time t and reference time r, where r ≤ t.
    Some fraction of samples have r = t (flow matching boundary condition).
    """
    
    def __init__(
        self,
        mean: float = -0.4,
        std: float = 1.0,
        data_proportion: float = 0.5,
    ):
        """Initialize mean flow time sampling.
        
        Args:
            mean: Mean for logit-normal distribution
            std: Standard deviation for logit-normal distribution
            data_proportion: Proportion of samples with r = t
        """
        self.mean = mean
        self.std = std
        self.data_proportion = data_proportion
    
    def sample_time(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Sample time t (returns t only, use sample_time_pair for both t and r)."""
        return logit_normal(key, (batch_size, 1), mean=self.mean, std=self.std, dtype=dtype)
    
    def sample_time_pair(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample time pair (t, r) with r ≤ t.
        
        Args:
            key: Random key
            batch_size: Batch size
            dtype: Output dtype
        
        Returns:
            Tuple of (t, r) where each is [B, 1]
        """
        return sample_tr(
            key,
            batch_size,
            dtype=dtype,
            mean=self.mean,
            std=self.std,
            data_proportion=self.data_proportion,
        )

