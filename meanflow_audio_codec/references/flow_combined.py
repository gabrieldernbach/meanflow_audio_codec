from dataclasses import dataclass
from enum import Enum
import torch, torch.nn as nn, torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from meanflow_audio_codec.datasets.mnist import load_mnist


class MethodType(Enum):
    """Flow method types."""
    FLOW_MATCHING = "flow_matching"
    MEAN_FLOW = "mean_flow"
    IMPROVED_MEAN_FLOW = "improved_mean_flow"


@dataclass
class Config:
    # Method selection
    method: str = "flow_matching"  # "flow_matching", "mean_flow", "improved_mean_flow"
    
    # Model architecture
    noise_dim: int = 28 * 28
    cond_dim: int = 512
    latent_dim: int = 1024
    n_blocks: int = 10
    n_classes: int = 10
    
    # Training
    batch_size: int = 512
    steps: int = 8_000
    warmup: int = 100
    device: str = 'mps'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    sample_n_steps: int = 100
    ema_beta: float = 0.99
    ema_alpha: float = 0.01
    log_frequency: int = 50
    eval_frequency: int = 200
    eval_steps: int = 100
    figsize: tuple[int, int] = (6, 6)
    
    # Flow matching specific
    noise_min: float = 0.001
    noise_max: float = 0.999
    
    # Mean Flow / Improved Mean Flow specific
    flow_ratio: float = 0.5
    gamma: float = 0.5
    c: float = 1e-3

    def __post_init__(self):
        """Set method-specific defaults (only override if still at Flow Matching defaults)."""
        if self.method in ("mean_flow", "improved_mean_flow"):
            if self.learning_rate == 1e-3:
                self.learning_rate = 1e-4
            if self.sample_n_steps == 100:
                self.sample_n_steps = 2
            if self.ema_beta == 0.99:
                self.ema_beta = 0.999
            if self.ema_alpha == 0.01:
                self.ema_alpha = 0.001


def sinusoidal_embedding(x, dim):
    freqs = torch.logspace(0, torch.log10(torch.tensor(1_000.)), dim//2).to(x.device)
    angles = 2*torch.pi*freqs[:, None] * x[None, :]
    emb = torch.cat((angles.sin(), angles.cos()), 0)   # [dim, B]
    return emb.T                                       # [B, dim]


def ema(mu, dx, beta, alpha): 
    return mu * beta + dx * alpha if mu is not None else dx


def mlp(ins, hidden, outs):
    return nn.Sequential(nn.Linear(ins, hidden), nn.SiLU(), nn.Linear(hidden, outs))


class ConditionalResidualBlock(nn.Module):
    """Residual block with adaptive layer normalization feature-wise modulation.
    
    See https://arxiv.org/pdf/2212.09748 for details.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cond = mlp(cfg.cond_dim, cfg.latent_dim, 3*cfg.noise_dim)
        self.mlp = mlp(cfg.noise_dim, cfg.latent_dim, cfg.noise_dim)
        self.n_blocks = cfg.n_blocks
    
    def forward(self, x, cond):
        res = x
        x = F.layer_norm(x, [x.shape[-1]])
        scale1, shift, scale2 = self.cond(cond).chunk(3, dim=-1)
        x = self.mlp((1+scale1)*x + shift) * (1+scale2)
        return res + x/self.n_blocks


class ConditionalFlow(nn.Module):
    """Unified conditional flow model supporting Flow Matching, Mean Flow, and Improved Mean Flow."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.method = cfg.method
        self.noise_dim = cfg.noise_dim
        self.blocks = nn.ModuleList([ConditionalResidualBlock(cfg) for _ in range(cfg.n_blocks)])
        self.cls_emb = nn.Sequential(
            nn.Embedding(cfg.n_classes, cfg.latent_dim), nn.SiLU(),
            nn.Linear(cfg.latent_dim, cfg.cond_dim)
        )

    def forward(self, x, time, cls_idx, ref_time=None):
        """Forward pass supporting both single-time (Flow Matching) and dual-time (Mean Flow) modes.
        
        Args:
            x: Input tensor [B, D]
            time: Time tensor [B, 1] or [B]
            cls_idx: Class indices [B]
            ref_time: Reference time [B, 1] or [B] (optional, for Mean Flow methods)
        """
        cls = self.cls_emb(cls_idx)
        
        if ref_time is not None:
            # Dual-time mode (Mean Flow / Improved Mean Flow)
            t_emb = sinusoidal_embedding(time.squeeze(1) if time.dim() > 1 else time, cls.size(-1))
            r_emb = sinusoidal_embedding(ref_time.squeeze(1) if ref_time.dim() > 1 else ref_time, cls.size(-1))
            cond = cls + t_emb + r_emb
        else:
            # Single-time mode (Flow Matching)
            t_emb = sinusoidal_embedding(time.squeeze(1) if time.dim() > 1 else time, cls.size(-1))
            cond = cls + t_emb
        
        for blk in self.blocks:
            x = blk(x, cond)
        return x

    def flow_matching_loss(self, x, cls_idx, noise_min, noise_max):
        """Flow Matching loss. See https://arxiv.org/pdf/2210.02747 equation (23)."""
        noise = torch.randn_like(x)
        time = torch.rand(size=(len(x), 1), device=x.device).sigmoid()
        noised = (1-time) * x + (noise_min + noise_max*time) * noise
        pred = self.forward(noised, time, cls_idx)
        return F.mse_loss(pred, noise.mul(noise_max).sub(x))

    def mean_flow_loss(self, x0, cls_idx, flow_ratio, gamma, c):
        """Mean Flow loss with adaptive reweighting."""
        B, D = x0.shape
        # sample (t, r) with r ≤ t and overwrite flow_ratio fraction with r=t
        t = torch.rand(B, device=x0.device)
        r = torch.rand(B, device=x0.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        same_mask = torch.rand(B, device=x0.device) < flow_ratio
        r = torch.where(same_mask, t, r)

        e = torch.randn_like(x0)                      # Gaussian end-point
        z = (1-t)[:, None]*x0 + t[:, None]*e          # mixture point
        v = e - x0

        def f(z_, t_, r_): 
            return self.forward(z_, t_.unsqueeze(1), cls_idx, r_.unsqueeze(1))
            
        u, dudt = torch.autograd.functional.jvp(
            f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)), create_graph=True)

        u_tgt = v - torch.clip((t-r)[:, None], min=0.0, max=1.0) * dudt
        err = u - u_tgt.detach()

        delta_sq = err.pow(2).mean(1)
        w = 1 / (delta_sq + c).pow(1-gamma)
        loss = (w.detach() * delta_sq).mean()

        return loss, err.pow(2).mean()

    def improved_mean_flow_loss(self, x0, cls_idx, flow_ratio):
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
        same_mask = torch.rand(B, device=x0.device) < flow_ratio
        r = torch.where(same_mask, t, r)

        e = torch.randn_like(x0)                      # Gaussian end-point
        z = (1-t)[:, None]*x0 + t[:, None]*e          # mixture point
        
        # Boundary condition: v_theta(z_t, t) = u_theta(z_t, t, t)
        v_theta = self.forward(z, t.unsqueeze(1), cls_idx, t.unsqueeze(1))
        
        def f(z_, t_, r_): 
            return self.forward(z_, t_.unsqueeze(1), cls_idx, r_.unsqueeze(1))
            
        # JVP w.r.t. (z, r, t) along tangent (v_theta, 0, 1)
        # Note: tangent vector is (v_theta, 0, 1) instead of (e-x, 1, 0)
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

    def loss(self, x, cls_idx, **kwargs):
        """Unified loss dispatcher based on method type."""
        if self.method == "flow_matching":
            return self.flow_matching_loss(x, cls_idx, self.cfg.noise_min, self.cfg.noise_max)
        elif self.method == "mean_flow":
            return self.mean_flow_loss(
                x, cls_idx, 
                kwargs.get("flow_ratio", self.cfg.flow_ratio),
                kwargs.get("gamma", self.cfg.gamma),
                kwargs.get("c", self.cfg.c)
            )
        elif self.method == "improved_mean_flow":
            return self.improved_mean_flow_loss(
                x, cls_idx,
                kwargs.get("flow_ratio", self.cfg.flow_ratio)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @torch.no_grad()
    def sample(self, cls_idx, n_steps=None):
        """Sample from the model using method-appropriate ODE solver."""
        n_steps = n_steps or self.cfg.sample_n_steps
        
        if self.method == "flow_matching":
            # Single-time sampling (Flow Matching)
            x = torch.randn(len(cls_idx), self.noise_dim, device=cls_idx.device)
            dt = 1.0 / n_steps
            for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
                t = t.expand(len(x), 1)
                k1 = self.forward(x, t, cls_idx)
                k2 = self.forward(x - dt*k1, t - dt, cls_idx)
                x = x - (dt / 2) * (k1 + k2)
            return x
        else:
            # Dual-time sampling (Mean Flow / Improved Mean Flow)
            B = len(cls_idx)
            x = torch.randn(B, self.noise_dim, device=cls_idx.device)
            t_vals = torch.linspace(1., 0., n_steps+1, device=x.device)
            for i in range(n_steps):
                t = t_vals[i].expand(B, 1)
                r = t_vals[i+1].expand(B, 1)
                dt = t - r
                k1 = self.forward(x, t, cls_idx, r)
                k2 = self.forward(x - dt*k1, r, cls_idx, r)
                x = x - (dt / 2) * (k1 + k2)
            return x


def init_training(cfg):
    """Initialize dataset iterators, model, and optimizer."""
    from pathlib import Path
    data_dir = str(Path.home() / "datasets" / "mnist")
    train_iterator = load_mnist(
        data_dir=data_dir,
        split='train',
        batch_size=cfg.batch_size,
        format='1d',
        normalize=True,
        seed=42
    )
    
    val_iterator = load_mnist(
        data_dir=data_dir,
        split='test',
        batch_size=cfg.batch_size,
        format='1d',
        normalize=True,
        seed=43  # Different seed for validation
    )
    
    model = ConditionalFlow(cfg).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    # Scheduler only for Flow Matching
    scheduler = None
    if cfg.method == "flow_matching":
        schedule = torch.cat([
            torch.linspace(0, 1, cfg.warmup),
            torch.logspace(0, -1, cfg.steps-cfg.warmup+1),
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=schedule.__getitem__
        )
    
    if scheduler is not None:
        return train_iterator, val_iterator, model, opt, scheduler
    else:
        return train_iterator, val_iterator, model, opt


@torch.no_grad()
def evaluate(model, val_iterator, cfg, n_steps):
    """Run evaluation pass on validation set."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    
    for _ in range(n_steps):
        img_np, lbl_np = next(val_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss_result = model.loss(img, lbl)
        
        if isinstance(loss_result, tuple):
            loss, mse = loss_result
            total_loss += loss.item()
            total_mse += mse.item()
        else:
            loss = loss_result
            total_loss += loss.item()
        
        n_batches += 1
    
    model.train()
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_mse = total_mse / n_batches if n_batches > 0 else 0.0
    
    return (avg_loss, avg_mse) if total_mse > 0 else avg_loss


def train(model, train_iterator, val_iterator, opt, cfg, scheduler=None):
    """Run training loop with periodic validation."""
    train_loss_ema = None
    val_loss_ema = None
    final_lbl = None
    
    for i in range(cfg.steps):
        # Training step
        img_np, lbl_np = next(train_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss_result = model.loss(img, lbl)
        
        if isinstance(loss_result, tuple):
            loss, mse = loss_result
        else:
            loss = loss_result
            mse = None

        loss.backward()
        opt.step()
        opt.zero_grad()
        if scheduler is not None:
            scheduler.step()

        train_loss_ema = ema(train_loss_ema, loss.item(), cfg.ema_beta, cfg.ema_alpha)
        
        # Periodic validation
        should_eval = (i + 1) % cfg.eval_frequency == 0
        if should_eval:
            eval_result = evaluate(model, val_iterator, cfg, cfg.eval_steps)
            if isinstance(eval_result, tuple):
                val_loss, val_mse = eval_result
            else:
                val_loss = eval_result
                val_mse = None
            val_loss_ema = ema(val_loss_ema, val_loss, cfg.ema_beta, cfg.ema_alpha)
        else:
            val_loss = None
            val_mse = None
        
        # Logging
        if i % cfg.log_frequency == 0:
            train_info = f'train_loss={loss.item():.6f} train_loss_ema={train_loss_ema:.6f}'
            val_info = f'val_loss_ema={val_loss_ema:.6f}' if val_loss_ema is not None else 'val_loss_ema=N/A'
            mse_info = f' mse={mse:.6f}' if mse is not None else ''
            print(f'{i=:05d}  {train_info}  {val_info}{mse_info}')
        elif should_eval:
            mse_info = f' val_mse={val_mse:.6f}' if val_mse is not None else ''
            print(f'{i=:05d}  eval: val_loss={val_loss:.6f} val_loss_ema={val_loss_ema:.6f}{mse_info}')
        
        final_lbl = lbl
    
    return final_lbl


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train flow model')
    parser.add_argument('--method', type=str, default='flow_matching',
                        choices=['flow_matching', 'mean_flow', 'improved_mean_flow'],
                        help='Flow method to use')
    args = parser.parse_args()
    
    cfg = Config(method=args.method)
    cfg.__post_init__()  # Apply method-specific defaults
    
    result = init_training(cfg)
    if len(result) == 5:
        train_iterator, val_iterator, model, opt, scheduler = result
        final_lbl = train(model, train_iterator, val_iterator, opt, cfg, scheduler)
    else:
        train_iterator, val_iterator, model, opt = result
        final_lbl = train(model, train_iterator, val_iterator, opt, cfg)
    
    smps = model.sample(final_lbl, n_steps=cfg.sample_n_steps)
    fig, axs = plt.subplots(4,4, figsize=cfg.figsize)
    for ax, xhat, idx in zip(axs.flatten(), smps[:16], final_lbl[:16]):
        ax.imshow(xhat.view(28,28).cpu(), vmin=-1, vmax=1, cmap='gray')
        ax.set_title(idx.item()); ax.axis('off')
    plt.tight_layout(); plt.show()

