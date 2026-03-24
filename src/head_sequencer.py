import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import hydra
import joblib
from omegaconf import DictConfig
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.dataloaders import NZDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM_HEAD(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim  = context_dim
        print("self.LSTM.input_dim: " , self.input_dim)

        self.W        = nn.Parameter(torch.randn(self.input_dim + 1 + hidden_dim, hidden_dim * 4) * 0.02)
        self.b        = nn.Parameter(torch.zeros(hidden_dim * 4))
        self.out_flow = nn.Linear(hidden_dim, 1)

    def cell_step(self, x_t, h_t, c_t):

        gates      = torch.cat((x_t, h_t), dim=-1) @ self.W + self.b
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        c_next     = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
        h_next     = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


    def forward_past(self, x_context, y_flow):
        """
        Encodeur série P pour l'INR — teacher forcing.
        x_context : (B, T, context_dim)  meteo + x_time
        y_flow    : (B, T, 1)
        Retourne  : hs (B, T, H), h_final, c_final
        """
        B, T, _ = x_context.shape
        h = torch.zeros(B, self.hidden_dim, device=x_context.device)
        c = torch.zeros(B, self.hidden_dim, device=x_context.device)
        hs = torch.zeros(B, T,self.hidden_dim, device=x_context.device)

        for t in range(T):
            x_in = torch.cat([
                x_context[:, t, :],
                y_flow[:, t, :],
            ], dim=-1)
            h, c = self.cell_step(x_in, h, c)

            hs[:,t,:] = h

        return hs, h, c


def warmup_cosine(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return lr_lambda


def compute_losses(preds, y, twin_idx, criterion):
    """
    preds  : (B, T-1, 1)  — prédit de t=1 à T
    y      : (B, T,   1)  — target complet
    twin_idx : frontière observation / horizon

    serie_p : loss sur [0 .. twin_idx-1]   — phase observation
    serie_h : loss sur [twin_idx-1 .. T-1] — phase horizon
    total   : loss sur toute la séquence
    """
    y_shifted = y[:, 1:, :]                         # (B, T-1, 1) — aligne avec preds

    loss_p = criterion(preds[:, :twin_idx - 1, :],  y_shifted[:, :twin_idx - 1, :])
    loss_h = criterion(preds[:, twin_idx - 1:, :],  y_shifted[:, twin_idx - 1:, :])
    loss   = criterion(preds,                        y_shifted)

    return loss, loss_p, loss_h


def run_epoch(model, loader, criterion, twin_idx, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total, total_p, total_h = 0.0, 0.0, 0.0

    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, leave=False, desc="train" if training else "val ")
        for batch in pbar:
            t_pts, _, _, x_meteo, x_time, y, _ = batch

            x_dyn = torch.cat([x_meteo, x_time], dim=-1).to(DEVICE)
            y     = y.to(DEVICE)

            preds            = model(x_dyn, y, twin_idx=twin_idx)
            loss, loss_p, loss_h = compute_losses(preds, y, twin_idx, criterion)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total   += loss.item()
            total_p += loss_p.item()
            total_h += loss_h.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", p=f"{loss_p.item():.4f}", h=f"{loss_h.item():.4f}")

    n = len(loader)
    return total / n, total_p / n, total_h / n


@hydra.main(version_base=None, config_path="../conf", config_name="head_lstm")
def main(cfg: DictConfig):
    conf = cfg.head_lstm
    print(f"Device : {DEVICE}")

    trainset = NZDataset(ROOT / conf.data.train_parquet, mode='train', latent_dim=conf.model.hidden_dim)
    valset   = NZDataset(ROOT / conf.data.val_parquet,   mode='val',   latent_dim=conf.model.hidden_dim, scalers=trainset.scalers)

    train_loader = DataLoader(trainset, batch_size=conf.data.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(valset,   batch_size=conf.data.batch_size, shuffle=False, num_workers=4)

    model     = LSTM_HEAD(conf.model.context_dim, conf.model.hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=conf.model.lr)
    criterion = nn.MSELoss()
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine(conf.model.warmup_epochs, conf.model.epochs))

    save_dir  = ROOT / 'save_models' / 'head_lstm'
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    joblib.dump(trainset.scalers, save_dir / 'scalers.pkl')
    print(f"Scalers sauvegardés : {save_dir / 'scalers.pkl'}")

    mlflow.set_experiment(conf.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            'hidden_dim':    conf.model.hidden_dim,
            'lr':            conf.model.lr,
            'warmup_epochs': conf.model.warmup_epochs,
            'batch_size':    conf.data.batch_size,
            'twin_idx':      conf.model.twin_idx,
            'device':        str(DEVICE),
        })

        run_name = mlflow.active_run().info.run_name
        best_val = float('inf')

        epochs_pbar = tqdm(range(conf.model.epochs), desc="Epochs")
        for epoch in epochs_pbar:
            tr, tr_p, tr_h = run_epoch(model, train_loader, criterion, conf.model.twin_idx, optimizer)
            va, va_p, va_h = run_epoch(model, val_loader,   criterion, conf.model.twin_idx)
            scheduler.step()

            lr = scheduler.get_last_lr()[0]
            mlflow.log_metrics({
                'train_loss':   tr,   'train_loss_p': tr_p, 'train_loss_h': tr_h,
                'val_loss':     va,   'val_loss_p':   va_p, 'val_loss_h':   va_h,
                'lr':           lr,
            }, step=epoch)

            if va < best_val:
                best_val  = va
                ckpt_path = save_dir / f"{run_name}_h{conf.model.hidden_dim}_{timestamp}_best.pt"
                torch.save({
                    'epoch':      epoch,
                    'model':      model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'scheduler':  scheduler.state_dict(),
                    'train_loss': tr, 'train_loss_p': tr_p, 'train_loss_h': tr_h,
                    'val_loss':   va, 'val_loss_p':   va_p, 'val_loss_h':   va_h,
                }, ckpt_path)

            epochs_pbar.set_postfix(
                tr=f"{tr:.4f}", tr_p=f"{tr_p:.4f}", tr_h=f"{tr_h:.4f}",
                va=f"{va:.4f}", va_p=f"{va_p:.4f}", va_h=f"{va_h:.4f}",
                lr=f"{lr:.2e}", best=f"{best_val:.4f}",
            )


if __name__ == "__main__":
    main()