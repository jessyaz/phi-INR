"""
baseline_lstm.py
────────────────────────────────────────────────────────────────────────────────
Baseline LSTM — trois modes :
  flow_only   : entrée = [y_flow]                        (dim = 1)
  static_only : entrée = [y_flow, static_emb]            (dim = 1 + STATIC_SEQ_DIM)
  dynamic     : entrée = [y_flow, static_emb, x_dyn]     (dim = 1 + STATIC_SEQ_DIM + x_dyn_dim)

Phase P (t = 0 .. twin_idx-2)   : teacher-forcing sur y observé
Phase H (t = twin_idx-1 .. T-2) : autorégressif
"""

from __future__ import annotations
from pathlib import Path
import sys

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
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.dataloaders import NZDataset, STAT_DIM, X_TIME_DIM

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOW_DIM       = 1
STATIC_SEQ_DIM = STAT_DIM + X_TIME_DIM + 8   # x_static + x_time + dir embedding


# ── Direction embedding ───────────────────────────────────────────────────────

class DirEmbedding(nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.embed = nn.Linear(1, output_dim)

    def forward(self, x):          # x : (B,)  → (B, output_dim)
        return self.embed(x.float().unsqueeze(-1))


# ── Baseline LSTM ─────────────────────────────────────────────────────────────

class BaselineLSTM(nn.Module):

    VALID_MODES = ("flow_only", "static_only", "dynamic")

    def __init__(
            self,
            hidden_dim: int = 128,
            mode:       str = "static_only",
            x_dyn_dim:  int = 0,
    ):
        super().__init__()
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.mode       = mode
        self.hidden_dim = hidden_dim

        # Encodeur statique (utilisé seulement si mode != flow_only)
        self.static_encoder = nn.Sequential(
            nn.Linear(STATIC_SEQ_DIM, STATIC_SEQ_DIM),
            nn.ReLU(),
            nn.Linear(STATIC_SEQ_DIM, STATIC_SEQ_DIM),
        )

        # Calcul de la dimension d'entrée du LSTM selon le mode
        if mode == "flow_only":
            lstm_input_dim = FLOW_DIM
        elif mode == "static_only":
            lstm_input_dim = FLOW_DIM + STATIC_SEQ_DIM
        else:  # dynamic
            lstm_input_dim = FLOW_DIM + STATIC_SEQ_DIM + x_dyn_dim

        self.lstm     = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, 1)

        print(
            f"[BaselineLSTM] mode={mode!r} | "
            f"STAT_DIM={STAT_DIM} X_TIME_DIM={X_TIME_DIM} "
            f"STATIC_SEQ_DIM={STATIC_SEQ_DIM} x_dyn_dim={x_dyn_dim} | "
            f"lstm_input={lstm_input_dim} | hidden={hidden_dim}"
        )

    # ── Construction de l'entrée à chaque pas ─────────────────────────────────

    def _build_input(self, flow, x_statics_t=None, x_dyn_t=None):
        """
        flow        : (B, 1)
        x_statics_t : (B, STATIC_SEQ_DIM)  — ignoré si flow_only
        x_dyn_t     : (B, x_dyn_dim)        — utilisé seulement si dynamic
        Retourne    : (B, 1, lstm_input_dim)
        """
        if self.mode == "flow_only":
            x = flow
        elif self.mode == "static_only":
            x = torch.cat([flow, self.static_encoder(x_statics_t.float())], dim=-1)
        else:  # dynamic
            x = torch.cat([flow, self.static_encoder(x_statics_t.float()), x_dyn_t], dim=-1)
        return x.unsqueeze(1)   # (B, 1, D)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, y_flow, twin_idx, x_statics=None, x_dyn=None):
        """
        y_flow    : (B, T, 1)
        x_statics : (B, T, STATIC_SEQ_DIM)  — requis si mode != flow_only
        x_dyn     : (B, T, x_dyn_dim)        — requis si mode == dynamic
        Retourne  : preds (B, T-1, 1)
        """
        B, T, _ = y_flow.shape
        device  = y_flow.device

        h = torch.zeros(1, B, self.hidden_dim, device=device)
        c = torch.zeros(1, B, self.hidden_dim, device=device)
        preds = []

        # Phase P : teacher-forcing
        for t in range(twin_idx - 1):
            x_in = self._build_input(
                y_flow[:, t, :],
                x_statics[:, t, :] if x_statics is not None else None,
                x_dyn[:, t, :]     if x_dyn     is not None else None,
            )
            out, (h, c) = self.lstm(x_in, (h, c))
            preds.append(self.out_proj(out.squeeze(1)))   # (B, 1)

        # Contexte dynamique figé au dernier pas connu (twin_idx - 1)
        x_dyn_fix = x_dyn[:, twin_idx - 1, :] if x_dyn is not None else None

        # Phase H : autorégressif
        last_pred = preds[-1]
        for t in range(twin_idx - 1, T - 1):
            x_in = self._build_input(
                last_pred,
                x_statics[:, t, :] if x_statics is not None else None,
                x_dyn_fix,
            )
            out, (h, c) = self.lstm(x_in, (h, c))
            last_pred   = self.out_proj(out.squeeze(1))
            preds.append(last_pred)

        return torch.stack(preds, dim=1)   # (B, T-1, 1)


# ── Pertes ────────────────────────────────────────────────────────────────────

def compute_losses(preds, y, twin_idx, criterion):
    y_shifted = y[:, 1:, :]
    loss_p = criterion(preds[:, :twin_idx - 1, :], y_shifted[:, :twin_idx - 1, :])
    loss_h = criterion(preds[:, twin_idx - 1:, :], y_shifted[:, twin_idx - 1:, :])
    loss   = criterion(preds, y_shifted)
    return loss, loss_p, loss_h


# ── LR schedule ───────────────────────────────────────────────────────────────

def warmup_cosine(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return lr_lambda


# ── Boucle entraînement / validation ─────────────────────────────────────────

def run_epoch(model, loader, criterion, twin_idx, dir_emb, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, total_p, total_h = 0.0, 0.0, 0.0

    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, leave=False, desc="train" if training else "val ")
        for batch in pbar:
            _, dir_idx, x_static, x_dyn, x_time, y, _ = batch

            y     = y.to(DEVICE)
            x_dyn = x_dyn.to(DEVICE)

            # ── x_statics séquentiel : (B, T, STATIC_SEQ_DIM) ────────────────
            dir_vec = dir_emb(dir_idx.to(DEVICE))                              # (B, 8)
            T       = x_time.size(1)
            x_statics_seq = torch.cat([
                x_static.unsqueeze(1).expand(-1, T, -1).to(DEVICE),
                x_time.to(DEVICE),
                dir_vec.unsqueeze(1).expand(-1, T, -1),
            ], dim=-1)                                                          # (B, T, STATIC_SEQ_DIM)

            preds = model(
                y_flow    = y,
                twin_idx  = twin_idx,
                x_statics = x_statics_seq if model.mode != "flow_only" else None,
                x_dyn     = x_dyn         if model.mode == "dynamic"   else None,
            )

            loss, loss_p, loss_h = compute_losses(preds, y, twin_idx, criterion)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total   += loss.item()
            total_p += loss_p.item()
            total_h += loss_h.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                p=f"{loss_p.item():.4f}",
                h=f"{loss_h.item():.4f}",
            )

    n = len(loader)
    return total / n, total_p / n, total_h / n


# ── Hydra ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../conf", config_name="baseline_lstm")
def main(cfg: DictConfig):
    conf = cfg.baseline_lstm
    print(f"Device  : {DEVICE}")
    print(f"Mode    : {conf.model.mode}")

    trainset = NZDataset(
        ROOT / conf.data.train_parquet,
        mode       = "train",
        latent_dim = conf.model.hidden_dim,
        )
    valset = NZDataset(
        ROOT / conf.data.val_parquet,
        mode       = "val",
        latent_dim = conf.model.hidden_dim,
        scalers    = trainset.scalers,
        )

    num_workers = conf.data.get("num_workers", 2)
    train_loader = DataLoader(
        trainset,
        batch_size         = conf.data.batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = num_workers > 0,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        valset,
        batch_size         = conf.data.batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = num_workers > 0,
        persistent_workers = num_workers > 0,
    )

    x_dyn_dim = trainset[0][3].shape[-1]   # dim de x_meteo

    dir_emb = DirEmbedding(output_dim=8).to(DEVICE)

    model = BaselineLSTM(
        hidden_dim = conf.model.hidden_dim,
        mode       = conf.model.mode,
        x_dyn_dim  = x_dyn_dim,
    ).to(DEVICE)

    optimizer = optim.Adam(
        list(model.parameters()) + list(dir_emb.parameters()),
        lr = conf.model.lr,
        )
    criterion = nn.MSELoss()
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine(conf.model.warmup_epochs, conf.model.epochs),
    )

    save_dir  = ROOT / "save_models" / f"baseline_lstm_{conf.model.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(trainset.scalers, save_dir / "scalers.pkl")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(conf.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"baseline_{conf.model.mode}"):
        mlflow.log_params({
            "model":      "BaselineLSTM",
            "mode":       conf.model.mode,
            "hidden_dim": conf.model.hidden_dim,
            "x_dyn_dim":  x_dyn_dim,
            "lr":         conf.model.lr,
            "batch_size": conf.data.batch_size,
            "twin_idx":   conf.model.twin_idx,
            "device":     str(DEVICE),
        })

        run_name = mlflow.active_run().info.run_name
        best_val = float("inf")

        for epoch in (pbar := tqdm(range(conf.model.epochs), desc="Epochs")):
            tr, tr_p, tr_h = run_epoch(
                model, train_loader, criterion,
                conf.model.twin_idx, dir_emb, optimizer,
            )
            va, va_p, va_h = run_epoch(
                model, val_loader, criterion,
                conf.model.twin_idx, dir_emb,
            )
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            mlflow.log_metrics({
                "train_loss":   tr,  "train_loss_p": tr_p, "train_loss_h": tr_h,
                "val_loss":     va,  "val_loss_p":   va_p, "val_loss_h":   va_h,
                "lr":           lr,
            }, step=epoch)

            if va < best_val:
                best_val  = va
                ckpt_path = save_dir / f"{run_name}_h{conf.model.hidden_dim}_{timestamp}_best.pt"
                torch.save({
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "dir_emb":      dir_emb.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "scheduler":    scheduler.state_dict(),
                    "mode":         conf.model.mode,
                    "val_loss":     va,  "val_loss_p":  va_p,  "val_loss_h":  va_h,
                    "train_loss":   tr,  "train_loss_p": tr_p, "train_loss_h": tr_h,
                }, ckpt_path)

            pbar.set_postfix(
                tr=f"{tr:.4f}", tr_p=f"{tr_p:.4f}", tr_h=f"{tr_h:.4f}",
                va=f"{va:.4f}", va_p=f"{va_p:.4f}", va_h=f"{va_h:.4f}",
                lr=f"{lr:.2e}", best=f"{best_val:.4f}",
            )

        print(f"\n[✓] Run terminé")
        print(f"    run_name : {run_name}")
        print(f"    best val : {best_val:.6f}")
        print(f"    ckpt     : {ckpt_path}")


if __name__ == "__main__":
    main()