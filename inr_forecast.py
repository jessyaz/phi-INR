from pathlib import Path
import os, sys, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import hydra
import joblib
import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from src.dataloaders import NZDataset, schema_fingerprint
from src.metalearning import outer_step
from src.network import ModulatedFourierFeatures
from src.snapshot import save_snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Construction du modèle
# ══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: DictConfig, device: torch.device) -> ModulatedFourierFeatures:
    return ModulatedFourierFeatures(
        input_dim        = cfg.data.input_dim,
        output_dim       = cfg.data.output_dim,
        look_back_window = cfg.data.look_back_window,
        num_frequencies  = cfg.inr.num_frequencies,
        latent_dim       = cfg.inr.latent_dim,
        lstm_hidden_dim  = cfg.inr.lstm_hidden_dim,
        spatial_dim      = cfg.inr.static.spatial_dim,
        dir_dim          = cfg.inr.static.dir_dim,
        num_directions   = cfg.inr.static.num_directions,
        sigma            = cfg.inr.static.sigma,
        width            = cfg.inr.hidden_dim,
        depth            = cfg.inr.depth,
        min_frequencies  = cfg.inr.min_frequencies,
        base_frequency   = cfg.inr.base_frequency,
        include_input    = cfg.inr.include_input,
        is_training      = True,
        use_context      = cfg.inr.use_context,
        freeze_lstm      = cfg.inr.get('freeze_lstm', False),
        control          = cfg.inr.get('control', None),
    ).to(device)


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
) -> tuple[float, float, float]:

    inr.train() if is_train else inr.eval()
    look_back   = cfg.data.look_back_window
    use_context = cfg.inr.use_context
    total, total_p, total_h, n = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, leave=False, desc="train" if is_train else "val  ")
        for batch in pbar:
            t, dir_idx, x_static, x_meteo, x_time, y, code = batch

            x_context = torch.cat([x_meteo, x_time], dim=-1).to(device)
            y_target  = y.to(device)
            t         = t.to(device)
            code      = code.to(device)

            x_statics = x_static.to(device) if use_context else None
            dir_idx   = dir_idx.to(device)  if use_context else None

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
                x_statics   = x_statics,
                dir_idx     = dir_idx,
            )

            if is_train:
                optimizer.zero_grad()
                outputs['loss'].backward()
                nn.utils.clip_grad_value_(inr.parameters(), cfg.optim.clip_grad_value)
                optimizer.step()

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

    return total / n, total_p / n, total_h / n


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="conf/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
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
        mode       = 'train',
        latent_dim = cfg.inr.latent_dim,
        )
    valset = NZDataset(
        ROOT / cfg.data.val_parquet,
        mode       = 'val',
        latent_dim = cfg.inr.latent_dim,
        scalers    = trainset.scalers,
        )

    train_loader = DataLoader(
        trainset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = cfg.data.num_workers,
        pin_memory         = cfg.data.pin_memory,
        persistent_workers = True,
        shuffle            = True,
    )
    val_loader = DataLoader(
        valset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = cfg.data.num_workers,
        pin_memory         = cfg.data.pin_memory,
        persistent_workers = True,
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
    optimizer = torch.optim.AdamW(
        inr.parameters(),
        lr           = cfg.optim.lr_inr,
        weight_decay = cfg.optim.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=cfg.optim.t_max)
        if cfg.optim.scheduler == "cosine" else None
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg.meta.experiment_pack)

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg.inr, resolve=True))
        mlflow.log_params({
            "look_back":  cfg.data.look_back_window,
            "horizon":    cfg.data.horizon,
            "lstm_in_dim":fp['lstm_in_dim'],
            "x_time_dim": fp['x_time_dim'],
            "time_feats": str(fp['time_features']),
        })
        run_name = mlflow.active_run().info.run_name

        # ── Dossier snapshot : save_models/snapshots/<experiment>/<run_name>/ ─
        run_dir = (
                ROOT / "save_models" / "snapshots"
                / cfg.meta.experiment_pack / run_name
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Snapshot dir : {run_dir}")

        # Scalers : dans le snapshot ET dans le chemin legacy de la config
        joblib.dump(trainset.scalers, run_dir / "scalers.pkl")
        scalers_legacy = ROOT / cfg.paths.scalers_file
        scalers_legacy.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trainset.scalers, scalers_legacy)
        print(f"Scalers      → {run_dir / 'scalers.pkl'}")

        best_val  = float('inf')
        ckpt_path = None

        for epoch in tqdm(range(cfg.optim.epochs), desc="Epochs"):
            tr, tr_p, tr_h = run_epoch(
                inr, train_loader, alpha, optimizer, cfg, device, is_train=True
            )
            va, va_p, va_h = run_epoch(
                inr, val_loader, alpha, None, cfg, device, is_train=False
            )

            if scheduler:
                scheduler.step()

            mlflow.log_metrics(
                {
                    #"lr":         scheduler.get_last_lr()[0],
                    'train_loss':   tr,  'train_loss_p': tr_p, 'train_loss_h': tr_h,
                    'val_loss':     va,  'val_loss_p':   va_p, 'val_loss_h':   va_h,
                },
                step=epoch,
            )

            # ── Sauvegarde du meilleur checkpoint ────────────────────────────
            if va < best_val:
                best_val  = va
                ckpt_path = run_dir / f"{run_name}_{timestamp}_best.pt"

                torch.save(
                    {
                        'epoch':           epoch,
                        'run_name':        run_name,
                        'timestamp':       timestamp,
                        # Configs complètes pour reconstruire le modèle
                        'cfg_inr':         OmegaConf.to_container(cfg.inr,   resolve=True),
                        'cfg_data':        OmegaConf.to_container(cfg.data,  resolve=True),
                        'cfg_inner':       OmegaConf.to_container(cfg.inner, resolve=True),
                        'cfg_optim':       OmegaConf.to_container(cfg.optim, resolve=True),
                        # Schéma de données au moment du run
                        'data_schema_fp':  fp,
                        # Poids
                        'inr_state_dict':  inr.state_dict(),
                        'optimizer':       optimizer.state_dict(),
                        # Métriques
                        'val_loss':        va,  'val_loss_p':   va_p,  'val_loss_h':  va_h,
                        'train_loss':      tr,  'train_loss_p': tr_p,  'train_loss_h':tr_h,
                        'static_encoder_type': type(inr.static_encoder).__name__,
                        'has_norm':            hasattr(inr.static_encoder, 'norm'),
                    },
                    ckpt_path,
                )

                # ── Snapshot : code + confs gelés ─────────────────────────────
                save_snapshot(run_dir=run_dir, root=ROOT, cfg=cfg)

            if epoch % cfg.misc.log_every_n_epochs == 0:
                tqdm.write(
                    f"[{epoch:>4}] "
                    f"tr={tr:.4f} tr_p={tr_p:.4f} tr_h={tr_h:.4f} | "
                    f"va={va:.4f} va_p={va_p:.4f} va_h={va_h:.4f} | "
                    f"best={best_val:.4f}"
                )

        # ── Résumé ────────────────────────────────────────────────────────────
        print(f"\n[✓] Run terminé")
        print(f"    run_name   : {run_name}")
        print(f"    best val   : {best_val:.6f}")
        if ckpt_path:
            print(f"    checkpoint : {ckpt_path}")
        print(f"    snapshot   : {run_dir}")


if __name__ == "__main__":
    main()