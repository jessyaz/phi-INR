from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import joblib
from tqdm import tqdm

from src.dataloaders import NZDataset
from src.metalearning import outer_step
from src.network import ModulatedFourierFeatures


# ── Helpers ──────────────────────────────────────────────────

def build_scheduler(optimizer, cfg):
    if cfg.optim.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.optim.t_max)
    if cfg.optim.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    return None


def build_model(cfg, device):
    inr = ModulatedFourierFeatures(
        input_dim        = cfg.data.input_dim,
        output_dim       = cfg.data.output_dim,
        x_dyn_c_dim      = cfg.data.x_dyn_c_dim,
        x_stat_dim       = cfg.data.x_stat_dim,
        look_back_window = cfg.data.look_back_window,
        num_frequencies  = cfg.inr.num_frequencies,
        latent_dim       = cfg.inr.latent_dim,
        static_emb_dim   = cfg.inr.static_emb_dim,
        width            = cfg.inr.hidden_dim,
        depth            = cfg.inr.depth,
        min_frequencies  = cfg.inr.min_frequencies,
        base_frequency   = cfg.inr.base_frequency,
        include_input    = cfg.inr.include_input,
        is_training      = True,
        use_context      = cfg.inr.use_context,
    )
    return inr.to(device)


# ── Main ─────────────────────────────────────────────────────

@hydra.main(config_path="conf/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    results_dir = ROOT / cfg.paths.save_models
    os.makedirs(results_dir, exist_ok=True)

    look_back = cfg.data.look_back_window
    horizon   = cfg.data.horizon
    print(f"Experiment : {cfg.meta.experiment_pack}")
    print(OmegaConf.to_yaml(cfg))

    # ── Dataset ──────────────────────────────────────────────
    trainset = NZDataset(
        parquet_file = ROOT / cfg.data.train_parquet,
        num_days     = cfg.data.num_days,
        latent_dim   = cfg.inr.latent_dim,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = cfg.data.num_workers,
        pin_memory         = cfg.data.pin_memory,
        persistent_workers = True,
    )
    joblib.dump(trainset.scalers, ROOT / cfg.paths.scalers_file)
    n_train = len(trainset)

    # ── Modèle ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    inr = build_model(cfg, device)

    vae_path = results_dir / "pretrain_vae" / "vae_best_model.pth"
    if vae_path.exists():
        print(f"Chargement VAE pré-entraîné : {vae_path}")
        inr.vae.load_state_dict(torch.load(vae_path, map_location=device))
        for param in inr.vae.parameters():
            param.requires_grad = True
    else:
        print("Aucun VAE pré-entraîné trouvé, entraînement from scratch.")

    # ── Optimiseur ───────────────────────────────────────────
    alpha     = nn.Parameter(torch.tensor([cfg.optim.lr_code], device=device))
    optimizer = torch.optim.AdamW(
        inr.parameters(),
        lr           = cfg.optim.lr_inr,
        weight_decay = cfg.optim.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    model_path = (
            results_dir / (
        f"models_forecasting_{cfg.data.dataset_name}_{horizon}"
        f"_{cfg.inr.use_context}_{cfg.meta.experiment_pack}"
        f"_{cfg.optim.epochs}_{cfg.misc.version}.pt"
    )
    )
    best_loss = np.inf

    # ── Entraînement ─────────────────────────────────────────
    for epoch in tqdm(range(cfg.optim.epochs)):

        beta = min(
            cfg.vae.beta_end,
            cfg.vae.beta_start
            + (cfg.vae.beta_end - cfg.vae.beta_start) * (epoch / cfg.vae.beta_warmup_epochs),
            )

        inr.train()
        fit_loss = 0.0

        for x_time, x_statics, x_dynamics, y_target, modulations in train_loader:
            x_time      = x_time.to(device)
            x_statics   = x_statics.to(device)
            x_dynamics  = x_dynamics.to(device)
            y_target    = y_target.to(device)
            modulations = modulations.to(device)

            coords_p  = x_time[:, :look_back, :]
            coords_h  = x_time[:, look_back:, :]
            features  = x_dynamics[:, :look_back, :]
            y_past    = y_target[:, :look_back, :]
            y_horizon = y_target[:, look_back:, :]
            n_samples = coords_p.shape[0]

            outputs = outer_step(
                func_rep    = inr,
                coords_p    = coords_p,
                coords_h    = coords_h,
                features_p  = features,
                x_statics   = x_statics,
                y_target_p  = y_past,
                y_target_h  = y_horizon,
                inner_steps = cfg.inner.inner_steps,
                inner_lr    = alpha,
                w_passed    = cfg.inner.w_passed,
                w_futur     = cfg.inner.w_futur,
                is_train    = True,
                modulations = torch.zeros_like(modulations),
                beta        = beta,
            )

            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=cfg.optim.clip_grad_value)
            optimizer.step()

            fit_loss += outputs["loss"].item() * n_samples

        train_loss = fit_loss / n_train

        if epoch % cfg.misc.log_every_n_epochs == 0:
            print(f"[epoch {epoch:>5}]  train_loss = {train_loss:.6f}  beta = {beta:.2e}")

        if scheduler is not None:
            scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss

        if epoch % cfg.misc.save_every_n_epochs == 0:
            torch.save(
                {
                    "cfg_inr":              OmegaConf.to_container(cfg.inr,  resolve=True),
                    "cfg_data":             OmegaConf.to_container(cfg.data, resolve=True),
                    "epoch":                epoch,
                    "inr_state_dict":       inr.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss":           train_loss,
                },
                model_path,
            )

    return train_loss


if __name__ == "__main__":
    main()