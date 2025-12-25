"""Flow Strategy Pattern Implementation

Implements three flow-based generative methods using the strategy pattern.

## Method Comparison

"""
| Aspect                  | Flow Matching     | Mean Flow        | Improved Mean Flow  |
|-------------------------|------------------|------------------|---------------------|
| Time Inputs             | t only           | t, r (r ≤ t)     | t, r (r ≤ t)        |
| Loss Type               | Direct regression| Adaptive weighted| Standard regression |
| Target Formulation      | e - x            | Network-dependent| Network-independent |
| JVP Direction           | N/A              | (e-x, 1, 0)      | (v_θ, 0, 1)         |
| Boundary Condition      | N/A              | Implicit         | Explicit v_θ = u_θ(t,t) |
| Reweighting             | No               | Yes (gamma, c)   | No                  |
| Sampling Steps          | ~100             | 2-5              | 2-5                 |
| Speedup vs FM           | 1×               | 20-50×           | 20-50×              |
| Training Stability      | High             | Medium           | High                |
| Implementation Complexity| Low              | Medium           | Medium              |
"""

## Flow Matching

**Key Characteristics**:
- Single-time `(x, t, cls_idx)` forward pass
- Direct regression: `v_θ(z_t, t) ≈ (e - x)` where `z_t = (1-t)x + t·e`
- Loss: Standard MSE on velocity prediction
- Noise schedule: Linear interpolation with `noise_min` and `noise_max`

**Advantages**: Simple, stable training, well-established baseline

**Disadvantages**: Slow generation (~100 steps), not suitable for real-time

**Reference**: Lipman et al., "Flow Matching for Generative Modeling", 2023
https://arxiv.org/pdf/2210.02747

## Mean Flow

**Key Characteristics**:
- Dual-time `(x, t, r, cls_idx)` with `r ≤ t`
- Models average velocity `u(z_t, r, t)` between timesteps `r` and `t`
- Adaptive reweighting: `w = 1/(error² + c)^(1-γ)` based on prediction error
- Hyperparameters: `gamma=0.5`, `c=1e-3`, `flow_ratio=0.5`

**Loss Function**:
- Sample `t, r` with `r ≤ t`, set `r=t` for `flow_ratio` fraction
- Compute `u_target = (e - x) - (t-r) * JVP(model, (z, t, r), (e-x, 1, 0))`
- Weighted loss: `loss = mean(w * error²)` where `error = u - detach(u_target)`

**Advantages**: Fast generation (2-5 steps), adaptive reweighting helps difficult samples

**Disadvantages**: Network-dependent target, more complex training objective

**Reference**: Geng et al., "Mean Flows for One-step Generative Modeling", 2025
https://arxiv.org/abs/2505.13447

## Improved Mean Flow

**Key Characteristics**:
- Dual-time `(x, t, r, cls_idx)` with `r ≤ t`
- Boundary condition: `v_θ(z_t, t) = u_θ(z_t, t, t)` (explicit velocity prediction)
- JVP uses `v_θ` instead of `(e - x)` in JVP computation
- Compound prediction: `V_θ = u_θ + (t-r) · detach(JVP)`
- Standard L2 regression (no adaptive weighting)
- Hyperparameters: `flow_ratio=0.5` only (simpler than Mean Flow)

**Loss Function**:
- Sample `t, r` with `r ≤ t`, set `r=t` for `flow_ratio` fraction
- Compute boundary: `v_theta = model(z, t, t, cls_idx)`
- JVP along `(v_theta, 0, 1)`: `u, dudt = JVP(model, (z, t, r), (v_theta, 0, 1))`
- Compound: `V_theta = u + (t-r) * detach(dudt)`
- Loss: `MSE(V_theta, e - x)`

**Advantages**: Network-independent target, simpler objective, more stable training,
same fast generation as Mean Flow (2-5 steps)

**Disadvantages**: Slightly more complex forward pass (boundary condition computation)

**Reference**: Geng et al., "Improved Mean Flows: On the Challenges of Fastforward Generative Models", 2024
https://arxiv.org/abs/2512.02012

## Method Progression

```
Flow Matching (baseline)
    ↓
Mean Flow (introduces average velocity, adaptive weighting)
    ↓
Improved Mean Flow (eliminates network-dependent target, stabilizes training)
```

**Recommendation**: Use Improved Mean Flow for best balance of speed, stability, and quality.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from tqdm import tqdm

from meanflow_audio_codec.references.model import ModelConfig

if TYPE_CHECKING:
    from meanflow_audio_codec.references.train import TrainConfig


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


@dataclass
class FlowStrategy:
    """Base strategy for flow methods."""
    name: str
    
    def loss_fn(self, model, x, cls_idx, flow_cfg: FlowStrategyConfig) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute loss. Returns single value or (loss, mse) tuple."""
        raise NotImplementedError
    
    def sample_fn(self, model, cls_idx, n_steps) -> torch.Tensor:
        """Generate samples."""
        raise NotImplementedError
    
    def make_config(self, **overrides) -> tuple[ModelConfig, TrainConfig, FlowStrategyConfig]:
        """Create method-specific configs with defaults. Returns (model_config, train_config, flow_strategy_config)."""
        raise NotImplementedError
    
    def create_scheduler(self, opt, train_cfg: "TrainConfig") -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup for all flow methods.
        
        Linear warmup from 0 to 1 over warmup steps, then stays at 1.0.
        No memory allocation - computes schedule on the fly.
        """
        warmup_steps = train_cfg.warmup
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=lr_lambda
        )
    
    def needs_ref_time(self) -> bool:
        """Whether forward pass needs ref_time parameter."""
        raise NotImplementedError
    
    def format_loss_log(self, loss, mse) -> str:
        """Format loss for logging. mse may be None."""
        if mse is not None:
            return f' mse={mse:.6f}'
        return ''


@dataclass
class FlowMatchingStrategy(FlowStrategy):
    """Strategy for Flow Matching method."""
    name: str = field(default="flow_matching", init=False)
    
    def loss_fn(self, model, x, cls_idx, flow_cfg: FlowStrategyConfig) -> torch.Tensor:
        """Flow Matching loss. See https://arxiv.org/pdf/2210.02747 equation (23)."""
        noise = torch.randn_like(x)
        time = torch.rand(size=(len(x), 1), device=x.device).sigmoid()
        noised = (1-time) * x + (flow_cfg.noise_min + flow_cfg.noise_max*time) * noise
        pred = model.forward(noised, time, cls_idx)
        return F.mse_loss(pred, noise.mul(flow_cfg.noise_max).sub(x))
    
    def sample_fn(self, model, cls_idx, n_steps) -> torch.Tensor:
        """Single-time sampling (Flow Matching)."""
        x = torch.randn(len(cls_idx), model.noise_dim, device=cls_idx.device)
        dt = 1.0 / n_steps
        for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
            t = t.expand(len(x), 1)
            k1 = model.forward(x, t, cls_idx)
            k2 = model.forward(x - dt*k1, t - dt, cls_idx)
            x = x - (dt / 2) * (k1 + k2)
        return x
    
    def make_config(self, **overrides) -> tuple[ModelConfig, TrainConfig, FlowStrategyConfig]:
        """Create Flow Matching configs with defaults."""
        model_cfg = ModelConfig()
        train_cfg = TrainConfig(
            learning_rate=1e-3,
            sample_n_steps=100,
            ema_beta=0.99,
            ema_alpha=0.01,
        )
        flow_cfg = FlowStrategyConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)
            elif hasattr(train_cfg, key):
                setattr(train_cfg, key, value)
            elif hasattr(flow_cfg, key):
                setattr(flow_cfg, key, value)
        
        return model_cfg, train_cfg, flow_cfg
    
    
    def needs_ref_time(self) -> bool:
        return False


@dataclass
class MeanFlowStrategy(FlowStrategy):
    """Strategy for Mean Flow method."""
    name: str = field(default="mean_flow", init=False)
    
    def loss_fn(self, model, x0, cls_idx, flow_cfg: FlowStrategyConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean Flow loss with adaptive reweighting."""
        B, D = x0.shape
        # sample (t, r) with r ≤ t and overwrite flow_ratio fraction with r=t
        t = torch.rand(B, device=x0.device)
        r = torch.rand(B, device=x0.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        same_mask = torch.rand(B, device=x0.device) < flow_cfg.flow_ratio
        r = torch.where(same_mask, t, r)

        e = torch.randn_like(x0)
        z = (1-t)[:, None]*x0 + t[:, None]*e
        v = e - x0

        def f(z_, t_, r_): 
            return model.forward(z_, t_.unsqueeze(1), cls_idx, r_.unsqueeze(1))
            
        u, dudt = torch.autograd.functional.jvp(
            f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)), create_graph=True)

        u_tgt = v - torch.clip((t-r)[:, None], min=0.0, max=1.0) * dudt
        err = u - u_tgt.detach()

        delta_sq = err.pow(2).mean(1)
        w = 1 / (delta_sq + flow_cfg.c).pow(1-flow_cfg.gamma)
        loss = (w.detach() * delta_sq).mean()

        return loss, err.pow(2).mean()
    
    def sample_fn(self, model, cls_idx, n_steps) -> torch.Tensor:
        """Dual-time sampling (Mean Flow)."""
        B = len(cls_idx)
        x = torch.randn(B, model.noise_dim, device=cls_idx.device)
        t_vals = torch.linspace(1., 0., n_steps+1, device=x.device)
        for i in range(n_steps):
            t = t_vals[i].expand(B, 1)
            r = t_vals[i+1].expand(B, 1)
            dt = t - r
            k1 = model.forward(x, t, cls_idx, r)
            k2 = model.forward(x - dt*k1, r, cls_idx, r)
            x = x - (dt / 2) * (k1 + k2)
        return x
    
    def make_config(self, **overrides) -> tuple[ModelConfig, TrainConfig, FlowStrategyConfig]:
        """Create Mean Flow configs with defaults."""
        model_cfg = ModelConfig()
        train_cfg = TrainConfig(
            learning_rate=1e-4,
            sample_n_steps=2,
            ema_beta=0.999,
            ema_alpha=0.001,
        )
        flow_cfg = FlowStrategyConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)
            elif hasattr(train_cfg, key):
                setattr(train_cfg, key, value)
            elif hasattr(flow_cfg, key):
                setattr(flow_cfg, key, value)
        
        return model_cfg, train_cfg, flow_cfg
    
    
    def needs_ref_time(self) -> bool:
        return True


@dataclass
class ImprovedMeanFlowStrategy(FlowStrategy):
    """Strategy for Improved Mean Flow method."""
    name: str = field(default="improved_mean_flow", init=False)
    
    def loss_fn(self, model, x0, cls_idx, flow_cfg: FlowStrategyConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Improved Mean Flow loss (iMF) using v-loss formulation.
        
        Key differences from original Mean Flow:
        1. Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
        2. JVP uses v_theta instead of e - x
        3. Compound prediction: V_theta = u_theta + (t-r) * sg(JVP)
        4. Loss: ||V_theta - (e - x)||^2 (standard regression)
        """
        B, D = x0.shape
        # sample (t, r) with r ≤ t and overwrite flow_ratio fraction with r=t
        t = torch.rand(B, device=x0.device)
        r = torch.rand(B, device=x0.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        same_mask = torch.rand(B, device=x0.device) < flow_cfg.flow_ratio
        r = torch.where(same_mask, t, r)

        e = torch.randn_like(x0)
        z = (1-t)[:, None]*x0 + t[:, None]*e
        
        # Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
        v_theta = model.forward(z, t.unsqueeze(1), cls_idx, t.unsqueeze(1))
        
        def f(z_, t_, r_): 
            return model.forward(z_, t_.unsqueeze(1), cls_idx, r_.unsqueeze(1))
            
        # JVP w.r.t. (z, r, t) along tangent (v_theta, 0, 1)
        u, dudt = torch.autograd.functional.jvp(
            f, (z, t, r), (v_theta, torch.zeros_like(t), torch.ones_like(t)), create_graph=True)
        
        # Compound prediction: V_theta = u_theta + (t-r) * sg(dudt)
        V_theta = u + (t-r)[:, None] * dudt.detach()
        
        # Target is ground truth conditional velocity
        target = e - x0
        
        # Standard L2 loss (no weighting)
        loss = (V_theta - target).pow(2).mean()
        mse = (V_theta - target).pow(2).mean()

        return loss, mse
    
    def sample_fn(self, model, cls_idx, n_steps) -> torch.Tensor:
        """Dual-time sampling (Improved Mean Flow)."""
        B = len(cls_idx)
        x = torch.randn(B, model.noise_dim, device=cls_idx.device)
        t_vals = torch.linspace(1., 0., n_steps+1, device=x.device)
        for i in range(n_steps):
            t = t_vals[i].expand(B, 1)
            r = t_vals[i+1].expand(B, 1)
            dt = t - r
            k1 = model.forward(x, t, cls_idx, r)
            k2 = model.forward(x - dt*k1, r, cls_idx, r)
            x = x - (dt / 2) * (k1 + k2)
        return x
    
    def make_config(self, **overrides) -> tuple[ModelConfig, TrainConfig, FlowStrategyConfig]:
        """Create Improved Mean Flow configs with defaults."""
        model_cfg = ModelConfig()
        train_cfg = TrainConfig(
            learning_rate=1e-4,
            sample_n_steps=2,
            ema_beta=0.999,
            ema_alpha=0.001,
        )
        flow_cfg = FlowStrategyConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)
            elif hasattr(train_cfg, key):
                setattr(train_cfg, key, value)
            elif hasattr(flow_cfg, key):
                setattr(flow_cfg, key, value)
        
        return model_cfg, train_cfg, flow_cfg
    
    def needs_ref_time(self) -> bool:
        return True


def get_strategy(method: str) -> FlowStrategy:
    """Get strategy by method name."""
    strategies = {
        "flow_matching": FlowMatchingStrategy(),
        "mean_flow": MeanFlowStrategy(),
        "improved_mean_flow": ImprovedMeanFlowStrategy(),
    }
    if method not in strategies:
        raise ValueError(f"Unknown method: {method}. Choose from {list(strategies.keys())}")
    return strategies[method]

