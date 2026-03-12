from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import joblib
from tqdm import tqdm

from src.dataloaders import NZDataset
from src.metalearning.metalearning_forecasting import outer_step
from src.network import ModulatedFourierFeatures


# ── Helpers ──────────────────────────────────────────────────

def build_scheduler(optimizer, cfg):
    if cfg.optim.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.optim.t_max)
    elif cfg.optim.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    return None


def build_model(cfg, device):
    inr = ModulatedFourierFeatures(
        input_dim           = cfg.data.input_dim,
        output_dim          = cfg.data.output_dim,
        x_dyn_c_dim         = cfg.data.x_dyn_c_dim,
        x_stat_dim          = cfg.data.x_stat_dim,
        look_back_window    = cfg.data.look_back_window,
        num_frequencies     = cfg.inr.num_frequencies,
        latent_dim          = cfg.inr.latent_dim,
        static_emb_dim      = cfg.inr.static_emb_dim,
        width               = cfg.inr.hidden_dim,
        depth               = cfg.inr.depth,
        modulate_scale      = cfg.inr.modulate_scale,
        modulate_shift      = cfg.inr.modulate_shift,
        frequency_embedding = cfg.inr.frequency_embedding,
        include_input       = cfg.inr.include_input,
        scale               = cfg.inr.scale,
        log_sampling        = cfg.inr.log_sampling,
        min_frequencies     = cfg.inr.min_frequencies,
        max_frequencies     = cfg.inr.max_frequencies,
        base_frequency      = cfg.inr.base_frequency,
        is_training         = True,
        use_context         = cfg.inr.use_context,
    )
    return inr.to(device)


# ── Main ─────────────────────────────────────────────────────

@hydra.main(config_path="../config/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    basepath    = str(ROOT)
    results_dir = str(ROOT / "save_models")
    import os; os.makedirs(results_dir, exist_ok=True)

    mlflow_db_path = str(ROOT / cfg.paths.mlflow_db)
    import os; os.makedirs(os.path.dirname(mlflow_db_path), exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    look_back_window = cfg.data.look_back_window
    horizon          = cfg.data.horizon

    experiment_pack = cfg.meta.experiment_pack
    print("Play name : ", experiment_pack)

    print(OmegaConf.to_yaml(cfg))

    # ── Dataset ──────────────────────────────────────────────
    trainset = NZDataset(
        parquet_file = cfg.data.train_parquet,
        num_days     = cfg.data.num_days,
        latent_dim   = cfg.inr.latent_dim,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size         = cfg.optim.batch_size,
        num_workers        = cfg.data.num_workers,
        pin_memory         = cfg.data.pin_memory,
        prefetch_factor    = cfg.data.prefetch_factor,
        persistent_workers = True,
    )
    joblib.dump(trainset.scalers, str(ROOT / cfg.paths.scalers_file))
    ntrain = len(trainset)

    # ── Modèle ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    inr = build_model(cfg, device)

    vae_path = str(ROOT / cfg.paths.vae_pretrain)
    if os.path.exists(vae_path):
        print(f"Chargement VAE pré-entraîné : {vae_path}")
        inr.vae.load_state_dict(torch.load(vae_path, map_location=device))

        for param in inr.vae.parameters():
            param.requires_grad = True

        #inr.vae.eval()

    else:
        print("Aucun VAE pré-entraîné trouvé, entraînement from scratch.")

    # ── Optimiseur ───────────────────────────────────────────
    alpha     = nn.Parameter(torch.tensor([cfg.optim.lr_code], device=device))
    optimizer = torch.optim.AdamW(
        [{"params": inr.parameters(), "lr": cfg.optim.lr_inr,
          "weight_decay": cfg.optim.weight_decay}],
        lr=cfg.optim.lr_inr, weight_decay=0,
    )
    scheduler = build_scheduler(optimizer, cfg)

    best_loss  = np.inf
    model_path = (
        f"{results_dir}/models_forecasting"
        f"_{cfg.data.dataset_name}_{horizon}_{cfg.inr.use_context}_{experiment_pack}"
        f"_{cfg.optim.epochs}_{cfg.misc.version}.pt"
    )

    # ── Entraînement ─────────────────────────────────────────
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for epoch in tqdm(range(cfg.optim.epochs)):

            beta = min(
                cfg.vae.beta_end,
                cfg.vae.beta_start + (cfg.vae.beta_end - cfg.vae.beta_start)
                * (epoch / cfg.vae.beta_warmup_epochs),
                )

            fit_train_loss = 0.0
            inr.train()

            for (x_time, x_statics, x_dynamics, y_target, modulations) in train_loader:

                x_time      = x_time.to(device)
                x_statics   = x_statics.to(device)
                x_dynamics  = x_dynamics.to(device)
                modulations = modulations.to(device)
                y_target    = y_target.to(device)

                coords_p   = x_time[:, :look_back_window, :]
                coords_h   = x_time[:, look_back_window:, :]
                context_p  = x_dynamics[:, :look_back_window, :]
                y_target_p = y_target[:, :look_back_window, :]
                y_target_h = y_target[:, look_back_window:, :]
                n_samples  = coords_p.shape[0]

                outputs = outer_step(
                    func_rep               = inr,
                    coordinates_p          = coords_p,
                    coordinates_h          = coords_h,
                    features_p             = context_p,
                    features_h             = context_p,
                    x_statics              = x_statics,
                    y_target_p             = y_target_p,
                    y_target_h             = y_target_h,
                    inner_steps            = cfg.inner.inner_steps,
                    inner_lr               = alpha,
                    look_back_window_size  = look_back_window,
                    horizon                = horizon,
                    w_passed               = cfg.inner.w_passed,
                    w_futur                = cfg.inner.w_futur,
                    lambda_vae             = cfg.vae.lambda_vae,
                    is_train               = True,
                    gradient_checkpointing = cfg.inner.gradient_checkpointing,
                    loss_type              = cfg.inner.loss_type,
                    lambda_fft             = cfg.inner.lambda_fft,
                    modulations            = torch.zeros_like(modulations),
                    beta                   = beta,
                )

                optimizer.zero_grad()
                outputs["loss"].backward(create_graph=False)
                nn.utils.clip_grad_value_(inr.parameters(), clip_value=cfg.optim.clip_grad_value)
                optimizer.step()

                fit_train_loss += outputs["loss"].item() * n_samples

            train_loss = fit_train_loss / ntrain

            if epoch % cfg.misc.log_every_n_epochs == 0:
                print(f"[epoch {epoch:>5}]  train_loss = {train_loss:.6f}  beta = {beta:.2e}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("beta_vae",   beta,       step=epoch)

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
                        "look_back_window":     look_back_window,
                        "horizon":              horizon,
                        "inr_state_dict":       inr.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss":           train_loss,
                    },
                    model_path,
                )

    return train_loss


if __name__ == "__main__":
    main()