from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from tqdm import tqdm

from meanflow_audio_codec.references.config import ModelConfig, TrainConfig, FlowStrategyConfig


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
    
    def create_scheduler(self, opt, train_cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup for all flow methods."""
        schedule = torch.cat([
            torch.linspace(0, 1, train_cfg.warmup),
            torch.logspace(0, -1, train_cfg.steps-train_cfg.warmup+1),
        ])
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=schedule.__getitem__
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

