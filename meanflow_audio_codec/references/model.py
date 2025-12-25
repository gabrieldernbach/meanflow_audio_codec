from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from meanflow_audio_codec.references.strategy import FlowStrategy


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    noise_dim: int = 28 * 28
    cond_dim: int = 512
    latent_dim: int = 1024
    n_blocks: int = 10
    n_classes: int = 10


def sinusoidal_embedding(x, dim):
    """Create sinusoidal positional embeddings.
    
    Args:
        x: Input tensor [B] or [B, 1]
        dim: Embedding dimension
    Returns:
        Embedding tensor [B, dim]
    """
    freqs = torch.logspace(0, torch.log10(torch.tensor(1_000.)), dim//2).to(x.device)
    angles = 2*torch.pi*freqs[:, None] * x[None, :]
    emb = torch.cat((angles.sin(), angles.cos()), 0)   # [dim, B]
    return emb.T                                       # [B, dim]


def mlp(ins, hidden, outs):
    """Create a simple MLP with SiLU activation."""
    return nn.Sequential(nn.Linear(ins, hidden), nn.SiLU(), nn.Linear(hidden, outs))


class ConditionalResidualBlock(nn.Module):
    """Residual block with adaptive layer normalization feature-wise modulation.
    
    See https://arxiv.org/pdf/2212.09748 for details.
    """
    def __init__(self, cfg: ModelConfig):
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
    """Unified conditional flow model using strategy pattern for different flow methods."""
    
    def __init__(self, model_cfg: ModelConfig, strategy: FlowStrategy):
        super().__init__()
        self.model_cfg = model_cfg
        self.strategy = strategy
        self.noise_dim = model_cfg.noise_dim
        self.blocks = nn.ModuleList([ConditionalResidualBlock(model_cfg) for _ in range(model_cfg.n_blocks)])
        self.cls_emb = nn.Sequential(
            nn.Embedding(model_cfg.n_classes, model_cfg.latent_dim), nn.SiLU(),
            nn.Linear(model_cfg.latent_dim, model_cfg.cond_dim)
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

    def loss(self, x, cls_idx, flow_cfg) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute loss using the injected strategy."""
        return self.strategy.loss_fn(self, x, cls_idx, flow_cfg)

    @torch.no_grad()
    def sample(self, cls_idx, n_steps=None, train_cfg=None) -> torch.Tensor:
        """Generate samples using the injected strategy."""
        n_steps = n_steps or (train_cfg.sample_n_steps if train_cfg else 100)
        return self.strategy.sample_fn(self, cls_idx, n_steps)

