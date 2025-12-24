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
    '''see https://arxiv.org/pdf/2212.09748 for adaptive layer norm feature-wise modulation'''
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

    def forward(self, x, time, cls_idx):
        cls = self.cls_emb(cls_idx)
        t_emb = sinusoidal_embedding(time.squeeze(1), cls.size(-1))
        for blk in self.blocks:
            x = blk(x, cls + t_emb)
        return x

    def loss(self, x, cls_idx, noise_min, noise_max):
        '''see https://arxiv.org/pdf/2210.02747 equation (23)'''
        noise = torch.randn_like(x)
        time = torch.rand(size=(len(x), 1), device=x.device).sigmoid()
        noised = (1-time) * x + (noise_min + noise_max*time) * noise
        pred = self.forward(noised, time, cls_idx)
        return F.mse_loss(pred, noise.mul(noise_max).sub(x))

    @torch.no_grad()
    def sample(self, cls_idx, n_steps=100):
        x = torch.randn(len(cls_idx), self.noise_dim, device=cls_idx.device)
        dt = 1.0 / n_steps
        for t in tqdm(torch.linspace(1, 0, n_steps, device=x.device)):
            t = t.expand(len(x), 1)
            k1 = self.forward(x, t, cls_idx)
            k2 = self.forward(x - dt*k1, t - dt, cls_idx)
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
    
    schedule = torch.cat([
        torch.linspace(0, 1, cfg.warmup),
        torch.logspace(0, -1, cfg.steps-cfg.warmup+1),
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=schedule.__getitem__
    )
    
    return train_iterator, val_iterator, model, opt, scheduler


@torch.no_grad()
def evaluate(model, val_iterator, cfg, n_steps):
    """Run evaluation pass on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for _ in range(n_steps):
        img_np, lbl_np = next(val_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss = model.loss(img, lbl, cfg.noise_min, cfg.noise_max)
        total_loss += loss.item()
        n_batches += 1
    
    model.train()
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def train(model, train_iterator, val_iterator, opt, scheduler, cfg):
    """Run training loop with periodic validation."""
    train_loss_ema = None
    val_loss_ema = None
    final_lbl = None
    
    for i in range(cfg.steps):
        # Training step
        img_np, lbl_np = next(train_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss = model.loss(img, lbl, cfg.noise_min, cfg.noise_max)

        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        train_loss_ema = ema(train_loss_ema, loss.item(), cfg.ema_beta, cfg.ema_alpha)
        
        # Periodic validation
        should_eval = (i + 1) % cfg.eval_frequency == 0
        if should_eval:
            val_loss = evaluate(model, val_iterator, cfg, cfg.eval_steps)
            val_loss_ema = ema(val_loss_ema, val_loss, cfg.ema_beta, cfg.ema_alpha)
        
        # Logging
        if i % cfg.log_frequency == 0:
            train_info = f'train_loss={loss.item():.6f} train_loss_ema={train_loss_ema:.6f}'
            val_info = f'val_loss_ema={val_loss_ema:.6f}' if val_loss_ema is not None else 'val_loss_ema=N/A'
            print(f'{i=:05d}  {train_info}  {val_info}')
        elif should_eval:
            print(f'{i=:05d}  eval: val_loss={val_loss:.6f} val_loss_ema={val_loss_ema:.6f}')
        
        final_lbl = lbl
    
    return final_lbl


if __name__ == '__main__':
    cfg = Config()
    
    train_iterator, val_iterator, model, opt, scheduler = init_training(cfg)
    final_lbl = train(model, train_iterator, val_iterator, opt, scheduler, cfg)
    
    smps = model.sample(final_lbl, n_steps=cfg.sample_n_steps)
    fig, axs = plt.subplots(4,4, figsize=cfg.figsize)
    for ax, xhat, idx in zip(axs.flatten(), smps[:16], final_lbl[:16]):
        ax.imshow(xhat.view(28,28).cpu(), vmin=-1, vmax=1, cmap='gray')
        ax.set_title(idx.item()); ax.axis('off')
    plt.tight_layout(); plt.show()

