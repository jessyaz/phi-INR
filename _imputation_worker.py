"""
_imputation_worker.py
=====================
Tourne dans un sous-process isolé — ne pas appeler directement.
Utilisé par evaluate_imputation.py.
"""
import sys
import argparse
import json as _json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot",       required=True)
parser.add_argument("--parquet",        required=True)
parser.add_argument("--missing-rate",   type=float, default=0.0)
parser.add_argument("--batch-size",     type=int,   default=32)
parser.add_argument("--num-workers",    type=int,   default=0)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--quiet",          action="store_true")
parser.add_argument("--no-site-filter", action="store_true")
args, _ = parser.parse_known_args()

SNAPSHOT_DIR = Path(args.snapshot).resolve()
sys.path.insert(0, str(SNAPSHOT_DIR / "code"))

import numpy as np
import torch
import joblib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders import NZDataset
from src.network import ModulatedFourierFeatures
from src.metalearning import outer_step


def log(msg):
    if not args.quiet:
        print(msg)


def compute_metrics(pred, target, eps=1e-8):
    mae   = (pred - target).abs().mean().item()
    rmse  = ((pred - target) ** 2).mean().sqrt().item()
    mape  = ((pred - target).abs() / (target.abs() + eps)).mean().item() * 100.0
    smape = (
                    2.0 * (pred - target).abs() / (pred.abs() + target.abs() + eps)
            ).mean().item() * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def build_model(cfg_inr, cfg_data, device):
    return ModulatedFourierFeatures(
        input_dim        = cfg_data.input_dim,
        output_dim       = cfg_data.output_dim,
        look_back_window = cfg_data.look_back_window,
        num_frequencies  = cfg_inr.num_frequencies,
        latent_dim       = cfg_inr.latent_dim,
        lstm_hidden_dim  = cfg_inr.lstm_hidden_dim,
        spatial_dim      = 4,
        dir_dim          = cfg_inr.static.dir_dim,
        num_directions   = cfg_inr.static.num_directions,
        sigma            = cfg_inr.static.sigma,
        width            = cfg_inr.hidden_dim,
        depth            = cfg_inr.depth,
        min_frequencies  = cfg_inr.min_frequencies,
        base_frequency   = cfg_inr.base_frequency,
        is_training      = False,
        use_context      = cfg_inr.use_context,
        freeze_lstm      = cfg_inr.get("freeze_lstm", False),
        control          = cfg_inr.get("control", None),
    ).to(device)


def build_context(x_statics, x_time, use_context, control, device):
    if not use_context:
        return None
    T = x_time.size(1)
    if control == "static_only":
        return x_statics.unsqueeze(1).repeat(1, T, 1).to(device)
    if control == "dynamic_only":
        return x_time.to(device)
    return torch.concat(
        [x_statics.unsqueeze(1).repeat(1, T, 1), x_time], dim=-1
    ).to(device)


#def forward_fill(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#    """(B, T, D) — remplace les timesteps masqués par la dernière valeur connue."""
#    out = tensor.clone()
#    for t in range(1, tensor.shape[1]):
#        if not mask[t]:
#            out[:, t, :] = out[:, t - 1, :]
#    return out


def forward_fill(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = tensor.clone()
    T_mask = mask.shape[0]   # = look_back
    for t in range(1, T_mask):
        if not mask[t]:
            out[:, t, :] = out[:, t - 1, :]
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    log(f"Snapshot     : {SNAPSHOT_DIR.name}")
    log(f"Missing rate : {args.missing_rate:.0%}")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_files = sorted(SNAPSHOT_DIR.glob("*_best.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"Aucun checkpoint dans {SNAPSHOT_DIR}")
    ckpt = torch.load(ckpt_files[-1], map_location=device)

    cfg_inr   = OmegaConf.create(ckpt["cfg_inr"])
    cfg_data  = OmegaConf.create(ckpt["cfg_data"])
    cfg_inner = OmegaConf.create(ckpt["cfg_inner"])
    cfg_optim = OmegaConf.create(ckpt["cfg_optim"])

    control     = cfg_inr.get("control", None)
    use_context = cfg_inr.use_context

    # ── Scalers ───────────────────────────────────────────────────────────────
    scalers  = joblib.load(SNAPSHOT_DIR / "scalers.pkl")
    scaler_t = scalers["target"]

    def inverse(tensor):
        shape = tensor.shape
        flat  = tensor.reshape(-1, 1).numpy()
        return torch.from_numpy(scaler_t.inverse_transform(flat)).reshape(shape)

    # ── Dataset ───────────────────────────────────────────────────────────────
    testset = NZDataset(
        args.parquet,
        mode             = "val",
        latent_dim       = cfg_inr.latent_dim,
        scalers          = scalers,
        skip_site_filter = args.no_site_filter,
    )
    if len(testset) == 0:
        raise RuntimeError("Dataset vide — essaie --no-site-filter")

    loader = DataLoader(testset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = build_model(cfg_inr, cfg_data, device)
    model.load_state_dict(ckpt["inr_state_dict"])
    model.eval()
    model._debug = False

    alpha = torch.tensor([ckpt.get("alpha", cfg_optim.lr_code)], device=device)
    look_back = cfg_data.look_back_window

    all_preds, all_targets = [], []

    # ── Inférence avec masquage ───────────────────────────────────────────────
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Imputation {args.missing_rate:.0%}", disable=args.quiet)
        for batch in pbar:
            t, dir_idx, x_statics, x_meteo, x_time, y, code = batch

            x_context = x_meteo.to(device)
            y_target  = y.to(device)
            t         = t.to(device)
            code      = code.to(device)

            # Masque reproductible
            mask    = torch.from_numpy(rng.random(look_back) > args.missing_rate)
            mask[0] = True

            # Forward fill sur le passé
            x_context_masked = forward_fill(x_context, mask)
            y_target_masked  = forward_fill(y_target,  mask)

            x_statics_in = build_context(x_statics, x_time, use_context, control, device)
            dir_idx_in   = dir_idx.to(device) if use_context else None

            coords_p    = t[:, :look_back, :]
            coords_h    = t[:, look_back:, :]
            x_context_p = x_context_masked[:, :look_back, :]
            x_context_h = torch.zeros_like( x_context[:, look_back:, :] )
            y_past      = y_target_masked[:, :look_back, :]
            y_horizon   = y_target[:, look_back:, :]

            outputs = outer_step(
                func_rep    = model,
                coords_p    = coords_p,
                coords_h    = coords_h,
                x_context_p = x_context_p,
                x_context_h = x_context_h,
                y_past      = y_past,
                y_horizon   = y_horizon,
                inner_steps = cfg_inner.inner_steps,
                inner_lr    = alpha,
                w_passed    = cfg_inner.w_passed,
                w_futur     = cfg_inner.w_futur,
                is_train    = False,
                code        = torch.zeros_like(code),
                x_statics   = x_statics_in,
                dir_idx     = dir_idx_in,
            )

            all_preds.append(outputs["out_h"].detach().cpu())
            all_targets.append(y_horizon.detach().cpu())

    preds   = inverse(torch.cat(all_preds,   dim=0))
    targets = inverse(torch.cat(all_targets, dim=0))
    metrics = compute_metrics(preds, targets)

    log(f"\n── Métriques (missing={args.missing_rate:.0%}) ──────────────")
    for k, v in metrics.items():
        log(f"  {k.upper():>6} : {v:.6f}")
    log("────────────────────────────────────────────────────")

    print("JSON:" + _json.dumps(metrics))


if __name__ == "__main__":
    main()