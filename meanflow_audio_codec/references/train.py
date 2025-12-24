import torch
import matplotlib.pyplot as plt

from meanflow_audio_codec.datasets.mnist import load_mnist
from meanflow_audio_codec.references.config import ModelConfig, TrainConfig, FlowStrategyConfig
from meanflow_audio_codec.references.strategy import FlowStrategy, get_strategy
from meanflow_audio_codec.references.model import ConditionalFlow


def ema(mu, dx, beta, alpha): 
    """Exponential moving average update."""
    return mu * beta + dx * alpha if mu is not None else dx


def init_training(cfg: Config, strategy: FlowStrategy):
    """Initialize dataset iterators, model, optimizer, and scheduler."""
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
    
    model = ConditionalFlow(cfg, strategy).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    scheduler = strategy.create_scheduler(opt, cfg)
    
    return train_iterator, val_iterator, model, opt, scheduler


@torch.no_grad()
def evaluate(model, val_iterator, cfg, strategy, n_steps):
    """Run evaluation pass on validation set."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    
    for _ in range(n_steps):
        img_np, lbl_np = next(val_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss_result = model.loss(img, lbl, cfg)
        
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


def train(model, train_iterator, val_iterator, opt, cfg, strategy, scheduler):
    """Run training loop with periodic validation."""
    train_loss_ema = None
    val_loss_ema = None
    final_lbl = None
    
    for i in range(cfg.steps):
        # Training step
        img_np, lbl_np = next(train_iterator)
        img = torch.from_numpy(img_np).to(cfg.device)
        lbl = torch.from_numpy(lbl_np).to(cfg.device)
        
        loss_result = model.loss(img, lbl, cfg)
        
        if isinstance(loss_result, tuple):
            loss, mse = loss_result
        else:
            loss = loss_result
            mse = None

        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        train_loss_ema = ema(train_loss_ema, loss.item(), cfg.ema_beta, cfg.ema_alpha)
        
        # Periodic validation
        should_eval = (i + 1) % cfg.eval_frequency == 0
        if should_eval:
            eval_result = evaluate(model, val_iterator, cfg, strategy, cfg.eval_steps)
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
            mse_val = mse.item() if mse is not None else None
            mse_info = strategy.format_loss_log(loss.item(), mse_val)
            print(f'{i=:05d}  {train_info}  {val_info}{mse_info}')
        elif should_eval:
            mse_info = strategy.format_loss_log(val_loss, val_mse)
            print(f'{i=:05d}  eval: val_loss={val_loss:.6f} val_loss_ema={val_loss_ema:.6f}{mse_info}')
        
        final_lbl = lbl
    
    return final_lbl


def main():
    """Main entry point for unified flow training reference implementation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train flow model')
    parser.add_argument('--method', type=str, default='flow_matching',
                        choices=['flow_matching', 'mean_flow', 'improved_mean_flow'],
                        help='Flow method to use')
    args = parser.parse_args()
    
    strategy = get_strategy(args.method)
    cfg = strategy.make_config()
    
    train_iterator, val_iterator, model, opt, scheduler = init_training(cfg, strategy)
    final_lbl = train(model, train_iterator, val_iterator, opt, cfg, strategy, scheduler)
    
    smps = model.sample(final_lbl, n_steps=cfg.sample_n_steps)
    fig, axs = plt.subplots(4,4, figsize=cfg.figsize)
    for ax, xhat, idx in zip(axs.flatten(), smps[:16], final_lbl[:16]):
        ax.imshow(xhat.view(28,28).cpu(), vmin=-1, vmax=1, cmap='gray')
        ax.set_title(idx.item()); ax.axis('off')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()

