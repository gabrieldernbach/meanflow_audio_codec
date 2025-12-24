"""Training step functions for flow models."""

import jax
import jax.numpy as jnp

from meanflow_audio_codec.models import TrainState
from meanflow_audio_codec.trainers.loss_strategies import (
    FlowMatchingLoss,
    ImprovedMeanFlowLoss,
    LossStrategy,
)
from meanflow_audio_codec.utils import logit_normal, sample_tr, weighted_l2_loss


def _train_step_with_strategy(
    state: TrainState,
    key: jax.random.PRNGKey,
    x: jnp.ndarray,
    loss_strategy: LossStrategy,
) -> tuple[TrainState, jnp.ndarray, jax.random.PRNGKey]:
    """Internal unified training step that uses a loss strategy.
    
    Args:
        state: Training state
        key: Random key
        x: Data samples [B, ...]
        loss_strategy: Loss strategy to use
    
    Returns:
        Updated state, loss value, new random key
    """
    loss, grads = loss_strategy.compute_loss(state, key, x)
    state = state.apply_gradients(grads=grads)
    return state, loss, key


def train_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    x: jnp.ndarray,
    loss_strategy: LossStrategy | None = None,
) -> tuple[TrainState, jnp.ndarray, jax.random.PRNGKey]:
    """Unified training step that uses a loss strategy.
    
    Args:
        state: Training state
        key: Random key
        x: Data samples [B, ...]
        loss_strategy: Loss strategy to use (default: FlowMatchingLoss)
    
    Returns:
        Updated state, loss value, new random key
    
    Note:
        Strategies are JAX-compatible and will be JIT-compiled when used
        in a JIT context. For best performance, create the strategy once
        outside the training loop and reuse it.
    """
    if loss_strategy is None:
        loss_strategy = FlowMatchingLoss()
    return _train_step_with_strategy(state, key, x, loss_strategy)


def train_step_improved_mean_flow(
    state: TrainState,
    key: jax.random.PRNGKey,
    x: jnp.ndarray,
) -> tuple[TrainState, jnp.ndarray, jax.random.PRNGKey]:
    """Improved Mean Flow training step (backward-compatible wrapper).
    
    This function is kept for backward compatibility. New code should use
    train_step() with an ImprovedMeanFlowLoss strategy.
    """
    return _train_step_improved_mean_flow_jit(state, key, x)

