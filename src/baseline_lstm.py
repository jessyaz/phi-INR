"""
baseline_lstm.py
────────────────────────────────────────────────────────────────────────────────
Baseline LSTM — 3 modes d'entrée, AR à partir de twin_idx.

Modes (model.mode) :
  dynamic_static  →  [x_meteo, x_time, y_flow, static_emb]   à chaque pas
  static_only     →  [y_flow, static_emb]                     à chaque pas
  flow_only       →  [y_flow]                                 à chaque pas

Phase P  (t = 0 .. twin_idx-2) : teacher-forcing sur y observé
Phase H  (t = twin_idx-1 .. T-2) : AR — le flow prédit alimente le pas suivant
                                    dynamic_static utilise x_meteo/x_time futurs

Sortie : preds (B, T-1, 1) — compatible avec compute_losses() de head_sequencer.py

Usage :
  python baseline_lstm.py baseline_lstm.model.mode=dynamic_static
  python baseline_lstm.py baseline_lstm.model.mode=static_only
  python baseline_lstm.py baseline_lstm.model.mode=flow_only
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

from src.dataloaders import NZDataset, METEO_DIM, X_TIME_DIM
from src.network import StaticEncoder

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOW_DIM = 1


# ── Baseline LSTM ─────────────────────────────────────────────────────────────

class BaselineLSTM(nn.Module):
    """
    Dimensions LSTM selon le mode :
      dynamic_static : METEO_DIM + X_TIME_DIM + FLOW_DIM + static_emb_dim
      static_only    : FLOW_DIM + static_emb_dim
      flow_only      : FLOW_DIM
    """

    VALID_MODES = ("dynamic_static", "static_only", "flow_only")

    def __init__(
            self,
            hidden_dim:     int   = 128,
            mode:           str   = "dynamic_static",
            spatial_dim:    int   = 32,
            dir_dim:        int   = 8,
            num_directions: int   = 6,
            sigma:          float = 0.1,
    ):
        super().__init__()
        assert mode in self.VALID_MODES, (
            f"mode doit être parmi {self.VALID_MODES}, reçu : {mode!r}"
        )
        self.hidden_dim = hidden_dim
        self.mode       = mode

        # ── Encodeur statique ─────────────────────────────────────────────────
        if mode in ("dynamic_static", "static_only"):
            self.static_encoder = StaticEncoder(
                spatial_dim    = spatial_dim,
                dir_dim        = dir_dim,
                num_directions = num_directions,
                sigma          = sigma,
            )
            static_emb_dim = spatial_dim + dir_dim
        else:
            static_emb_dim = 0

        self.static_emb_dim = static_emb_dim

        # ── Dimension d'entrée ────────────────────────────────────────────────
        if mode == "dynamic_static":
            lstm_input_dim = METEO_DIM + X_TIME_DIM + FLOW_DIM + static_emb_dim
        elif mode == "static_only":
            lstm_input_dim = FLOW_DIM + static_emb_dim
        else:
            lstm_input_dim = FLOW_DIM

        self.lstm_input_dim = lstm_input_dim
        self.lstm           = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.out_proj       = nn.Linear(hidden_dim, 1)

        print(
            f"[BaselineLSTM] mode={mode!r} | "
            f"lstm_input={lstm_input_dim} | hidden={hidden_dim}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_static_emb(self, x_statics, dir_idx):
        """(B, static_emb_dim) ou None."""
        if self.mode in ("dynamic_static", "static_only"):
            assert x_statics is not None and dir_idx is not None, \
                f"x_statics et dir_idx requis pour mode={self.mode!r}"
            return self.static_encoder(x_statics.float(), dir_idx)
        return None

    def _build_step(self, flow_t, x_dyn_t=None, static_emb=None):
        """
        Construit le vecteur d'entrée pour un seul pas de temps.
        flow_t     : (B, 1)
        x_dyn_t    : (B, METEO_DIM+X_TIME_DIM)  — utilisé si mode=dynamic_static
        static_emb : (B, static_emb_dim)         — utilisé si mode != flow_only
        Retourne   : (B, 1, lstm_input_dim)
        """
        if self.mode == "dynamic_static":
            x_in = torch.cat([x_dyn_t, flow_t, static_emb], dim=-1)
        elif self.mode == "static_only":
            x_in = torch.cat([flow_t, static_emb], dim=-1)
        else:
            x_in = flow_t
        return x_in.unsqueeze(1)   # (B, 1, D)

    # ── Forward : teacher-forcing phase P, AR phase H ─────────────────────────

    def forward(
            self,
            x_dyn,           # (B, T, METEO_DIM+X_TIME_DIM)
            y_flow,          # (B, T, 1)
            twin_idx: int,   # frontière observation / horizon
            x_statics=None,  # (B, STAT_DIM)
            dir_idx=None,    # (B,)
    ):
        """
        Retourne preds (B, T-1, 1).
          preds[:,  :twin_idx-1, :] ← phase P (teacher-forcing)
          preds[:, twin_idx-1:,  :] ← phase H (autorégressif depuis twin_idx)
        """
        B, T, _ = y_flow.shape
        device   = y_flow.device

        static_emb = self._get_static_emb(x_statics, dir_idx)

        h = torch.zeros(1, B, self.hidden_dim, device=device)
        c = torch.zeros(1, B, self.hidden_dim, device=device)

        preds = []

        # ── Phase P : t = 0 … twin_idx-2 → prédit t+1 = 1 … twin_idx-1 ──────
        for t in range(twin_idx - 1):
            x_in        = self._build_step(y_flow[:, t, :], x_dyn[:, t, :], static_emb)
            out, (h, c) = self.lstm(x_in, (h, c))
            preds.append(self.out_proj(out.squeeze(1)))   # (B, 1)

        # ── Phase H : AR depuis twin_idx ─────────────────────────────────────
        # Input flow = dernière prédiction ; x_dyn futur reste disponible
        last_pred = preds[-1]   # (B, 1)
        for t in range(twin_idx - 1, T - 1):
            x_in        = self._build_step(last_pred, x_dyn[:, t, :], static_emb)
            out, (h, c) = self.lstm(x_in, (h, c))
            last_pred   = self.out_proj(out.squeeze(1))
            preds.append(last_pred)

        # (B, T-1, 1)
        return torch.stack(preds, dim=1)#.unsqueeze(-1)


# ── Pertes ────────────────────────────────────────────────────────────────────

def compute_losses(preds, y, twin_idx, criterion):
    """Compatible avec head_sequencer.compute_losses."""
    y_shifted = y[:, 1:, :]
    loss_p = criterion(preds[:, :twin_idx - 1, :], y_shifted[:, :twin_idx - 1, :])
    loss_h = criterion(preds[:, twin_idx - 1:, :], y_shifted[:, twin_idx - 1:, :])
    loss   = criterion(preds, y_shifted)
    return loss, loss_p, loss_h


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

def warmup_cosine(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return lr_lambda


def run_epoch(model, loader, criterion, twin_idx, optimizer=None):
    training     = optimizer is not None
    needs_static = model.mode in ("dynamic_static", "static_only")
    model.train() if training else model.eval()

    total, total_p, total_h = 0.0, 0.0, 0.0

    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, leave=False, desc="train" if training else "val ")
        for batch in pbar:
            _, dir_idx, x_static, x_meteo, x_time, y, _ = batch

            x_dyn    = torch.cat([x_meteo, x_time], dim=-1).to(DEVICE)
            y        = y.to(DEVICE)
            x_static = x_static.to(DEVICE)
            dir_idx  = dir_idx.to(DEVICE)

            preds = model(
                x_dyn, y, twin_idx,
                x_statics = x_static if needs_static else None,
                dir_idx   = dir_idx  if needs_static else None,
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
    print(f"Device : {DEVICE}")
    print(f"Mode   : {conf.model.mode}")

    trainset = NZDataset(ROOT / conf.data.train_parquet, mode="train")
    valset   = NZDataset(ROOT / conf.data.val_parquet,   mode="val", scalers=trainset.scalers)

    train_loader = DataLoader(trainset, batch_size=conf.data.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(valset,   batch_size=conf.data.batch_size, shuffle=False, num_workers=4)

    model = BaselineLSTM(
        hidden_dim     = conf.model.hidden_dim,
        mode           = conf.model.mode,
        spatial_dim    = conf.model.get("spatial_dim",    32),
        dir_dim        = conf.model.get("dir_dim",         8),
        num_directions = conf.model.get("num_directions",  6),
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=conf.model.lr)
    criterion = nn.MSELoss()
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine(conf.model.warmup_epochs, conf.model.epochs))

    save_dir  = ROOT / "save_models" / f"baseline_lstm_{conf.model.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(trainset.scalers, save_dir / "scalers.pkl")

    mlflow.set_experiment(conf.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"baseline_{conf.model.mode}"):
        mlflow.log_params({
            "model":      "BaselineLSTM",
            "mode":       conf.model.mode,
            "hidden_dim": conf.model.hidden_dim,
            "lr":         conf.model.lr,
            "batch_size": conf.data.batch_size,
            "twin_idx":   conf.model.twin_idx,
            "device":     str(DEVICE),
        })

        run_name = mlflow.active_run().info.run_name
        best_val = float("inf")

        epochs_pbar = tqdm(range(conf.model.epochs), desc="Epochs")
        for epoch in epochs_pbar:
            tr, tr_p, tr_h = run_epoch(model, train_loader, criterion, conf.model.twin_idx, optimizer)
            va, va_p, va_h = run_epoch(model, val_loader,   criterion, conf.model.twin_idx)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            mlflow.log_metrics({
                "train_loss": tr, "train_loss_p": tr_p, "train_loss_h": tr_h,
                "val_loss":   va, "val_loss_p":   va_p, "val_loss_h":   va_h,
                "lr":         lr,
            }, step=epoch)

            if va < best_val:
                best_val  = va
                ckpt_path = save_dir / f"{run_name}_h{conf.model.hidden_dim}_{timestamp}_best.pt"
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "mode":       conf.model.mode,
                    "train_loss": tr, "train_loss_p": tr_p, "train_loss_h": tr_h,
                    "val_loss":   va, "val_loss_p":   va_p, "val_loss_h":   va_h,
                }, ckpt_path)

            epochs_pbar.set_postfix(
                tr=f"{tr:.4f}", tr_p=f"{tr_p:.4f}", tr_h=f"{tr_h:.4f}",
                va=f"{va:.4f}", va_p=f"{va_p:.4f}", va_h=f"{va_h:.4f}",
                lr=f"{lr:.2e}", best=f"{best_val:.4f}",
            )


if __name__ == "__main__":
    main()