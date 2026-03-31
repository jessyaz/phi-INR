from pathlib import Path
import os, sys, warnings, random, math
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import hydra
import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from src.dataloaders import NZDataset, schema_fingerprint
from src.metalearning import outer_step
from src.network import ModulatedFourierFeatures
from src.snapshot import save_snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Reproductibilité
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] {seed}")


# ══════════════════════════════════════════════════════════════════════════════
# Construction du modèle
# ══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: DictConfig, device: torch.device) -> ModulatedFourierFeatures:
    print(f"[Model] spatial_dim={cfg.inr.static.spatial_dim}  control={cfg.inr.get('control', None)}")
    return ModulatedFourierFeatures(
        input_dim        = cfg.data.input_dim,
        output_dim       = cfg.data.output_dim,
        look_back_window = cfg.data.look_back_window,
        num_frequencies  = cfg.inr.num_frequencies,
        latent_dim       = cfg.inr.latent_dim,
        lstm_hidden_dim  = cfg.inr.lstm_hidden_dim,
        spatial_dim      = cfg.inr.static.spatial_dim,   # ← toujours 128
        dir_dim          = cfg.inr.static.dir_dim,
        num_directions   = cfg.inr.static.num_directions,
        sigma            = cfg.inr.static.sigma,
        width            = cfg.inr.hidden_dim,
        depth            = cfg.inr.depth,
        min_frequencies  = cfg.inr.min_frequencies,
        base_frequency   = cfg.inr.base_frequency,
        is_training      = True,
        use_context      = cfg.inr.use_context,
        freeze_lstm      = cfg.inr.get('freeze_lstm', False),
        control          = cfg.inr.get('control', None),
    ).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Construction du vecteur de contexte statique
# ══════════════════════════════════════════════════════════════════════════════

def build_context(
        x_statics:   torch.Tensor,   # (B, STAT_DIM)
        x_time:      torch.Tensor,   # (B, T, X_TIME_DIM)
        use_context: bool,
        device:      torch.device,
) -> torch.Tensor | None:
    """
    Retourne toujours le concat geo+time (dim 4) au StaticEncoder.
    Le routing ablation est géré par control dans LatentToModulation.
    """
    if not use_context:
        return None
    T = x_time.size(1)
    return torch.concat(
        [x_statics.unsqueeze(1).repeat(1, T, 1), x_time], dim=-1
    ).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled sampling TF → AR
# ══════════════════════════════════════════════════════════════════════════════

def _tf_ratio(epoch: int, total_epochs: int, warmup: int = 10) -> float:
    if epoch < warmup:
        return 1.0
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════════════
# Métriques
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
        pred:   torch.Tensor,
        target: torch.Tensor,
        eps:    float = 1e-8,
) -> dict[str, float]:
    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mape = ((pred - target).abs() / (target.abs() + eps)).mean().item() * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ══════════════════════════════════════════════════════════════════════════════
# Époque d'entraînement / validation
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
        inr:       nn.Module,
        loader:    DataLoader,
        alpha:     nn.Parameter,
        optimizer: torch.optim.Optimizer | None,
        cfg:       DictConfig,
        device:    torch.device,
        is_train:  bool = True,
        tf_ratio:  float = 0.0,
) -> tuple[float, float, float, dict[str, float]]:

    inr.train() if is_train else inr.eval()
    look_back   = cfg.data.look_back_window
    use_context = cfg.inr.use_context

    total, total_p, total_h, n = 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, leave=False, desc="train" if is_train else "val  ")
        for batch in pbar:
            t, dir_idx, x_statics, x_meteo, x_time, y, code = batch

            x_context = x_meteo.to(device)
            y_target  = y.to(device)
            t         = t.to(device)
            code      = code.to(device)

            x_statics_in = build_context(
                x_statics   = x_statics.to(device),
                x_time      = x_time.to(device),
                use_context = use_context,
                device      = device,
            )
            dir_idx_in = dir_idx.to(device) if use_context else None

            coords_p    = t[:, :look_back, :]
            coords_h    = t[:, look_back:, :]
            x_context_p = x_context[:, :look_back, :]
            x_context_h = x_context[:, look_back:, :]
            y_past      = y_target[:, :look_back, :]
            y_horizon   = y_target[:, look_back:, :]

            outputs = outer_step(
                func_rep    = inr,
                coords_p    = coords_p,
                coords_h    = coords_h,
                x_context_p = x_context_p,
                x_context_h = x_context_h,
                y_past      = y_past,
                y_horizon   = y_horizon,
                inner_steps = cfg.inner.inner_steps,
                inner_lr    = alpha,
                w_passed    = cfg.inner.w_passed,
                w_futur     = cfg.inner.w_futur,
                is_train    = is_train,
                code        = torch.zeros_like(code),
                x_statics   = x_statics_in,
                dir_idx     = dir_idx_in,
                tf_ratio    = tf_ratio if is_train else 0.0,
            )

            if is_train:
                optimizer.zero_grad()
                outputs['loss'].backward()
                nn.utils.clip_grad_value_(inr.parameters(), cfg.optim.clip_grad_value)
                optimizer.step()
            else:
                all_preds.append(outputs['out_h'].detach().cpu())
                all_targets.append(y_horizon.detach().cpu())

            bs       = coords_p.shape[0]
            total   += outputs['loss'].item()   * bs
            total_p += outputs['loss_p'].item() * bs
            total_h += outputs['loss_h'].item() * bs
            n       += bs
            pbar.set_postfix(
                loss=f"{outputs['loss'].item():.4f}",
                p=f"{outputs['loss_p'].item():.4f}",
                h=f"{outputs['loss_h'].item():.4f}",
            )

    val_metrics: dict[str, float] = {}
    if not is_train and all_preds:
        preds       = torch.cat(all_preds,   dim=0)
        targets     = torch.cat(all_targets, dim=0)
        val_metrics = compute_metrics(preds, targets)

    return total / n, total_p / n, total_h / n, val_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="conf/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):

    set_seed(cfg.meta.get("seed", 42))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    print(f"Device      : {device}")
    print(f"use_context : {cfg.inr.use_context}")
    print(f"control     : {cfg.inr.get('control', None)}")

    # ── Schéma de données ─────────────────────────────────────────────────────
    fp = schema_fingerprint()
    print(f"[DataSchema] meteo={fp['meteo_dim']}  x_time={fp['x_time_dim']}  "
          f"lstm_in={fp['lstm_in_dim']}  feats={fp['time_features']}")

    # ── Cohérence avec head_lstm.yaml ─────────────────────────────────────────
    lstm_cfg = OmegaConf.load(ROOT / 'conf' / 'head_lstm.yaml').head_lstm
    assert cfg.data.look_back_window == lstm_cfg.model.twin_idx, (
        f"look_back_window ({cfg.data.look_back_window}) != twin_idx "
        f"({lstm_cfg.model.twin_idx})"
    )
    assert cfg.inr.lstm_hidden_dim == lstm_cfg.model.hidden_dim, (
        f"lstm_hidden_dim ({cfg.inr.lstm_hidden_dim}) != hidden_dim "
        f"({lstm_cfg.model.hidden_dim})"
    )
    print(f"[✓] Cohérence : look_back={cfg.data.look_back_window}  "
          f"horizon={cfg.data.horizon}  lstm_hidden={cfg.inr.lstm_hidden_dim}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    trainset = NZDataset(
        ROOT / cfg.data.train_parquet,
        mode             = 'train',
        latent_dim       = cfg.inr.latent_dim,
        skip_site_filter = False,
        )
    valset = NZDataset(
        ROOT / cfg.data.val_parquet,
        mode             = 'val',
        latent_dim       = cfg.inr.latent_dim,
        scalers          = trainset.scalers,
        skip_site_filter = False,
        )

    num_workers = cfg.data.num_workers
    train_loader = DataLoader(
        trainset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = num_workers,
        pin_memory         = cfg.data.pin_memory if num_workers > 0 else False,
        persistent_workers = num_workers > 0,
        shuffle            = True,
    )
    val_loader = DataLoader(
        valset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = num_workers,
        pin_memory         = cfg.data.pin_memory if num_workers > 0 else False,
        persistent_workers = num_workers > 0,
        shuffle            = False,
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    inr = build_model(cfg, device)
    inr._debug = False
    print(f"Paramètres  : {sum(p.numel() for p in inr.parameters()):,}")

    lstm_ckpt = cfg.inr.get('lstm_ckpt', None)
    if lstm_ckpt:
        inr.load_lstm_weights(ROOT / lstm_ckpt, device=device)
    else:
        print("LSTM initialisé from scratch.")

    alpha     = nn.Parameter(torch.tensor([cfg.optim.lr_code], device=device))
    optimizer = torch.optim.AdamW(inr.parameters(), lr=cfg.optim.lr_inr)

    def warmup_cosine(warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        return lr_lambda

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine(
            cfg.optim.get("warmup_epochs", 10),
            cfg.optim.get("epochs", 100),
        )
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(cfg.meta.experiment_pack)

    forced_run_name = cfg.meta.get("run_name", None)

    with mlflow.start_run(run_name=forced_run_name):
        mlflow.log_params(OmegaConf.to_container(cfg.inr, resolve=True))
        mlflow.log_params({
            "look_back":  cfg.data.look_back_window,
            "horizon":    cfg.data.horizon,
            "lstm_in_dim":fp['lstm_in_dim'],
            "x_time_dim": fp['x_time_dim'],
            "time_feats": str(fp['time_features']),
            "seed":       cfg.meta.get("seed", 42),
        })
        run_name = mlflow.active_run().info.run_name

        run_dir = (
                ROOT / "save_models" / "snapshots"
                / cfg.meta.experiment_pack / run_name
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Snapshot dir : {run_dir}")

        joblib.dump(trainset.scalers, run_dir / "scalers.pkl")
        scalers_legacy = ROOT / cfg.paths.scalers_file
        scalers_legacy.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trainset.scalers, scalers_legacy)
        print(f"Scalers      → {run_dir / 'scalers.pkl'}")

        best_val  = float('inf')
        ckpt_path = None

        for epoch in tqdm(range(cfg.optim.epochs), desc="Epochs"):

            ratio = _tf_ratio(
                epoch,
                total_epochs = cfg.optim.epochs,
                warmup       = cfg.optim.get("warmup_epochs", 20),
            )

            tr, tr_p, tr_h, _           = run_epoch(
                inr, train_loader, alpha, optimizer,
                cfg, device, is_train=True, tf_ratio=ratio,
            )
            va, va_p, va_h, val_metrics = run_epoch(
                inr, val_loader, alpha, None,
                cfg, device, is_train=False, tf_ratio=0.0,
            )

            if scheduler:
                scheduler.step()

            mlflow.log_metrics(
                {
                    "tf_ratio":     ratio,
                    "lr":           scheduler.get_last_lr()[0],
                    'train_loss':   tr,  'train_loss_p': tr_p, 'train_loss_h': tr_h,
                    'val_loss':     va,  'val_loss_p':   va_p, 'val_loss_h':   va_h,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                },
                step=epoch,
            )

            if va < best_val:
                best_val  = va
                ckpt_path = run_dir / f"{run_name}_{timestamp}_best.pt"

                torch.save(
                    {
                        'epoch':               epoch,
                        'run_name':            run_name,
                        'timestamp':           timestamp,
                        'cfg_inr':             OmegaConf.to_container(cfg.inr,   resolve=True),
                        'cfg_data':            OmegaConf.to_container(cfg.data,  resolve=True),
                        'cfg_inner':           OmegaConf.to_container(cfg.inner, resolve=True),
                        'cfg_optim':           OmegaConf.to_container(cfg.optim, resolve=True),
                        'data_schema_fp':      fp,
                        'alpha':               alpha.item(),
                        'inr_state_dict':      inr.state_dict(),
                        'optimizer':           optimizer.state_dict(),
                        'val_loss':            va,  'val_loss_p':   va_p,  'val_loss_h':  va_h,
                        'train_loss':          tr,  'train_loss_p': tr_p,  'train_loss_h':tr_h,
                        'val_mae':             val_metrics.get('mae'),
                        'val_rmse':            val_metrics.get('rmse'),
                        'val_mape':            val_metrics.get('mape'),
                        'static_encoder_type': type(inr.static_encoder).__name__,
                        'has_norm':            hasattr(inr.static_encoder, 'norm'),
                        'seed':                cfg.meta.get("seed", 42),
                    },
                    ckpt_path,
                )

                save_snapshot(run_dir=run_dir, root=ROOT, cfg=cfg)

            if epoch % cfg.misc.log_every_n_epochs == 0:
                metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                tqdm.write(
                    f"[{epoch:>4}] "
                    f"tr={tr:.4f} tr_p={tr_p:.4f} tr_h={tr_h:.4f} | "
                    f"va={va:.4f} va_p={va_p:.4f} va_h={va_h:.4f} | "
                    f"best={best_val:.4f} | {metrics_str}"
                )

        print(f"\n[✓] Run terminé")
        print(f"    run_name   : {run_name}")
        print(f"    best val   : {best_val:.6f}")
        if ckpt_path:
            print(f"    checkpoint : {ckpt_path}")
        print(f"    snapshot   : {run_dir}")


if __name__ == "__main__":
    main()