from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from meanflow_audio_codec.datasets.mnist import load_mnist

@dataclass
class Config:
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
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    flow_ratio: float = 0.5
    gamma: float = 0.5
    c: float = 1e-3
    sample_n_steps: int = 2
    ema_beta: float = 0.999
    ema_alpha: float = 0.001
    log_frequency: int = 50
    eval_frequency: int = 200
    eval_steps: int = 100
    figsize: tuple[int, int] = (6, 6)

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
    def __init__(self, cfg):
        super().__init__()
        self.noise_dim = cfg.noise_dim
        self.blocks = nn.ModuleList([ConditionalResidualBlock(cfg) for _ in range(cfg.n_blocks)])
        self.cls_emb = nn.Sequential(
            nn.Embedding(cfg.n_classes, cfg.latent_dim), nn.SiLU(),
            nn.Linear(cfg.latent_dim, cfg.cond_dim)
        )

    def forward(self, x, t, r, cls_idx):
        cls = self.cls_emb(cls_idx)
        t_emb = sinusoidal_embedding(t.squeeze(1), cls.size(-1))
        r_emb = sinusoidal_embedding(r.squeeze(1), cls.size(-1))
        cond  = cls + t_emb + r_emb
        for blk in self.blocks:
            x = blk(x, cond)
        return x

    def mean_flow_loss(self, x0, cls_idx, flow_ratio, gamma, c):
        B, D = x0.shape
        # sample (t, r) with r ≤ t and overwrite flow_ratio fraction with r=t
        t = torch.rand(B, device=x0.device)
        r = torch.rand(B, device=x0.device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        same_mask = torch.rand(B, device=x0.device) < flow_ratio
        r = torch.where(same_mask, t, r)

        e  = torch.randn_like(x0)                      # Gaussian end-point
        z  = (1-t)[:, None]*x0 + t[:, None]*e          # mixture point
        v  = e - x0

        def f(z_, t_, r_): 
            return self.forward(z_, t_.unsqueeze(1), r_.unsqueeze(1), cls_idx)
            
        u, dudt = torch.autograd.functional.jvp(
            f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)), create_graph=True)

        u_tgt = v - torch.clip((t-r)[:, None], min=0.0, max=1.0) * dudt
        err   = u - u_tgt.detach()

        delta_sq = err.pow(2).mean(1)
        w = 1 / (delta_sq + c).pow(1-gamma)
        loss = (w.detach() * delta_sq).mean()

        return loss, err.pow(2).mean()

    @torch.no_grad()
    def sample(self, cls_idx, n_steps=5):
        B = len(cls_idx)
        x = torch.randn(B, self.noise_dim, device=cls_idx.device)
        t_vals = torch.linspace(1., 0., n_steps+1, device=x.device)
        for i in range(n_steps):
            t = t_vals[i].expand(B, 1)
            r = t_vals[i+1].expand(B, 1)
            dt = t - r
            k1 = self.forward(x, t, r, cls_idx)
            k2 = self.forward(x - dt*k1, r, r, cls_idx)
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
        
        loss, mse = model.mean_flow_loss(
            img, lbl, cfg.flow_ratio, cfg.gamma, cfg.c
        )
        
        total_loss += loss.item()
        total_mse += mse.item()
        n_batches += 1
    
    model.train()
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_mse = total_mse / n_batches if n_batches > 0 else 0.0
    return avg_loss, avg_mse


def train(model, train_iterator, val_iterator, opt, cfg):
    """Run training loop with periodic validation."""
    train_loss_ema = None
    val_loss_ema = None
    final_lbl = None
    
    for i in range(cfg.steps):
        # Training step
        img_np, lbl_np = next(train_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss, mse = model.mean_flow_loss(
            img, lbl, cfg.flow_ratio, cfg.gamma, cfg.c
        )

        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss_ema = ema(train_loss_ema, loss.item(), cfg.ema_beta, cfg.ema_alpha)
        
        # Periodic validation
        should_eval = (i + 1) % cfg.eval_frequency == 0
        if should_eval:
            val_loss, val_mse = evaluate(model, val_iterator, cfg, cfg.eval_steps)
            val_loss_ema = ema(val_loss_ema, val_loss, cfg.ema_beta, cfg.ema_alpha)
        
        # Logging
        if i % cfg.log_frequency == 0:
            train_info = f'train_loss={loss.item():.6f} train_loss_ema={train_loss_ema:.6f}'
            val_info = f'val_loss_ema={val_loss_ema:.6f}' if val_loss_ema is not None else 'val_loss_ema=N/A'
            print(f'{i=:05d}  {train_info}  {val_info}  mse={mse:.6f}')
        elif should_eval:
            print(f'{i=:05d}  eval: val_loss={val_loss:.6f} val_loss_ema={val_loss_ema:.6f} val_mse={val_mse:.6f}')
        
        final_lbl = lbl
    
    return final_lbl


def main():
    """Main entry point for MeanFlow reference implementation."""
    cfg = Config()
    
    train_iterator, val_iterator, model, opt = init_training(cfg)
    final_lbl = train(model, train_iterator, val_iterator, opt, cfg)
    
    smps = model.sample(final_lbl, n_steps=cfg.sample_n_steps)
    fig, axs = plt.subplots(4,4, figsize=cfg.figsize)
    for ax, xhat, idx in zip(axs.flatten(), smps[:16], final_lbl[:16]):
        ax.imshow(xhat.view(28,28).cpu(), vmin=-1, vmax=1, cmap='gray')
        ax.set_title(idx.item()); ax.axis('off')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
