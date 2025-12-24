"""Loss strategies for flow matching training.

Loss strategies encapsulate the complete loss computation logic,
including noise sampling, time sampling, target computation, and loss calculation.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from meanflow_audio_codec.models import TrainState
from meanflow_audio_codec.trainers.noise_schedules import (
    LinearNoiseSchedule,
    NoiseSchedule,
    UniformNoiseSchedule,
)
from meanflow_audio_codec.trainers.time_sampling import (
    LogitNormalTimeSampling,
    MeanFlowTimeSampling,
    TimeSamplingStrategy,
    UniformTimeSampling,
)
from meanflow_audio_codec.utils import weighted_l2_loss


class LossStrategy(ABC):
    """Abstract base class for loss computation strategies."""
    
    @abstractmethod
    def compute_loss(
        self,
        state: TrainState,
        key: jax.random.PRNGKey,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Compute loss and gradients.
        
        Args:
            state: Training state
            key: Random key
            x: Data samples [B, ...]
        
        Returns:
            Tuple of (loss, grads)
        """
        pass


class FlowMatchingLoss(LossStrategy):
    """Standard flow matching loss strategy.
    
    Uses single time sampling and standard velocity prediction.
    """
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule | None = None,
        time_sampling: TimeSamplingStrategy | None = None,
        use_weighted_loss: bool = True,
    ):
        """Initialize flow matching loss strategy.
        
        Args:
            noise_schedule: Noise schedule (default: LinearNoiseSchedule)
            time_sampling: Time sampling strategy (default: LogitNormalTimeSampling)
            use_weighted_loss: If True, use weighted_l2_loss; else use MSE
        """
        self.noise_schedule = noise_schedule or LinearNoiseSchedule()
        self.time_sampling = time_sampling or LogitNormalTimeSampling()
        self.use_weighted_loss = use_weighted_loss
    
    def compute_loss(
        self,
        state: TrainState,
        key: jax.random.PRNGKey,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Compute flow matching loss."""
        key, k_noise, k_time = jax.random.split(key, 3)
        
        # Sample noise
        noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
        
        # Sample time
        time = self.time_sampling.sample_time(k_time, x.shape[0], dtype=x.dtype)
        
        # Prepare time for model [B, 2] with h=0
        time_pair = jnp.concatenate([time, jnp.zeros_like(time)], axis=-1)
        
        # Interpolate
        noised = self.noise_schedule.interpolate(x, noise, time)
        
        # Compute target
        target = self.noise_schedule.compute_target(x, noise)
        
        def loss_fn(params):
            # Encode clean data to get latent
            latents = state.apply_fn({"params": params}, x, method="encode")
            
            # Predict velocity
            pred = state.apply_fn({"params": params}, noised, time_pair, latents)
            
            # Compute loss
            if self.use_weighted_loss:
                return weighted_l2_loss(pred, target)
            else:
                delta = pred - target
                return jnp.mean(delta ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, grads


class MeanFlowLoss(LossStrategy):
    """Mean flow loss strategy with adaptive reweighting.
    
    Uses dual time sampling (t, r) and adaptive reweighting based on prediction error.
    """
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule | None = None,
        time_sampling: MeanFlowTimeSampling | None = None,
        gamma: float = 0.5,
        c: float = 1e-3,
    ):
        """Initialize mean flow loss strategy.
        
        Args:
            noise_schedule: Noise schedule (default: LinearNoiseSchedule)
            time_sampling: Time sampling strategy (default: MeanFlowTimeSampling)
            gamma: Reweighting exponent (default: 0.5)
            c: Stability constant for reweighting (default: 1e-3)
        """
        self.noise_schedule = noise_schedule or LinearNoiseSchedule()
        self.time_sampling = time_sampling or MeanFlowTimeSampling()
        self.gamma = gamma
        self.c = c
    
    def compute_loss(
        self,
        state: TrainState,
        key: jax.random.PRNGKey,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Compute mean flow loss with adaptive reweighting."""
        key, k_noise, k_tr = jax.random.split(key, 3)
        
        # Sample noise
        noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
        
        # Sample time pair (t, r)
        time, r = self.time_sampling.sample_time_pair(k_tr, x.shape[0], dtype=x.dtype)
        
        # Interpolate: z = (1-t) * x + t * e
        # For mean flow, we use uniform interpolation (not scaled)
        if time.ndim == 1:
            time = time[:, None]
        noised = (1.0 - time) * x + time * noise
        
        # Target velocity: v = e - x
        target = noise - x
        
        def loss_fn(params):
            # Encode clean data to get latent
            latents = state.apply_fn({"params": params}, x, method="encode")
            
            # Define u function for JVP
            def u_fn(z_local, t_local, r_local):
                h_local = t_local - r_local
                th = jnp.concatenate([t_local, h_local], axis=-1)
                return state.apply_fn({"params": params}, z_local, th, latents)
            
            # Compute JVP: (u, dudt) = JVP of u_fn at (z, t, r) in direction (v, 1, 0)
            # where v = e - x
            (u, dudt) = jax.jvp(
                u_fn,
                (noised, time, r),
                (target, jnp.ones_like(time), jnp.zeros_like(r)),
            )
            
            # Mean flow target: u_target = v - clip(t-r) * dudt
            t_minus_r = jnp.clip(time - r, 0.0, 1.0)
            u_target = target - t_minus_r * jax.lax.stop_gradient(dudt)
            
            # Error: err = u - detach(u_target)
            err = u - u_target
            
            # Adaptive reweighting: w = 1 / (error² + c)^(1-γ)
            # Compute per-example squared error
            delta_sq = jnp.mean(err ** 2, axis=tuple(range(1, err.ndim)))
            w = jax.lax.stop_gradient(1.0 / (delta_sq + self.c) ** (1.0 - self.gamma))
            
            # Weighted loss: mean(w * error²)
            loss = jnp.mean(w * delta_sq)
            
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, grads


class ImprovedMeanFlowLoss(LossStrategy):
    """Improved mean flow loss strategy.
    
    Uses dual time sampling (t, r) and JVP with network-independent target.
    """
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule | None = None,
        time_sampling: MeanFlowTimeSampling | None = None,
        use_weighted_loss: bool = True,
    ):
        """Initialize improved mean flow loss strategy.
        
        Args:
            noise_schedule: Noise schedule (default: LinearNoiseSchedule)
            time_sampling: Time sampling strategy (default: MeanFlowTimeSampling)
            use_weighted_loss: If True, use weighted_l2_loss; else use MSE
        """
        self.noise_schedule = noise_schedule or LinearNoiseSchedule()
        self.time_sampling = time_sampling or MeanFlowTimeSampling()
        self.use_weighted_loss = use_weighted_loss
    
    def compute_loss(
        self,
        state: TrainState,
        key: jax.random.PRNGKey,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jax.Array]:
        """Compute improved mean flow loss."""
        key, k_noise, k_tr = jax.random.split(key, 3)
        
        # Sample noise
        noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
        
        # Sample time pair (t, r)
        time, r = self.time_sampling.sample_time_pair(k_tr, x.shape[0], dtype=x.dtype)
        
        # Interpolate
        noised = self.noise_schedule.interpolate(x, noise, time)
        
        # Compute target
        target = self.noise_schedule.compute_target(x, noise)
        
        def loss_fn(params):
            # Encode clean data to get latent
            latents = state.apply_fn({"params": params}, x, method="encode")
            
            # Define u function for JVP
            def u_fn(z_local, t_local, r_local):
                h_local = t_local - r_local
                th = jnp.concatenate([t_local, h_local], axis=-1)
                return state.apply_fn({"params": params}, z_local, th, latents)
            
            # Get initial velocity prediction at (x_t, t, t)
            t_pair = jnp.concatenate([time, jnp.zeros_like(time)], axis=-1)
            v = state.apply_fn({"params": params}, noised, t_pair, latents)
            
            # Compute JVP: (u, dudt) = JVP of u_fn at (x_t, time, r) in direction (v, 1, 0)
            (u, dudt) = jax.jvp(
                u_fn,
                (noised, time, r),
                (v, jnp.ones_like(time), jnp.zeros_like(r)),
            )
            
            # Improved mean flow prediction: v_pred = u + (t - r) * stop_gradient(dudt)
            v_pred = u + (time - r) * jax.lax.stop_gradient(dudt)
            
            # Compute loss
            if self.use_weighted_loss:
                return weighted_l2_loss(v_pred, target)
            else:
                delta = v_pred - target
                return jnp.mean(delta ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, grads

