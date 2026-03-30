"""
evaluate.py — figé au moment du snapshot.
Ce fichier est copié tel quel dans chaque run_dir/.

Usage:
    python evaluate.py --parquet /chemin/vers/test.parquet
    python evaluate.py --parquet /chemin/vers/test.parquet --batch-size 64 --quiet
"""
import sys
import argparse
import json as _json
from pathlib import Path

# ── Ajout du code figé au path AVANT tout import projet ──────────────────────
SNAPSHOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SNAPSHOT_DIR / "code"))

import torch
import joblib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders import NZDataset
from src.network import ModulatedFourierFeatures
from src.metalearning import outer_step


# ══════════════════════════════════════════════════════════════════════════════
# Métriques
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> dict:
    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mape = ((pred - target).abs() / (target.abs() + eps)).mean().item() * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ══════════════════════════════════════════════════════════════════════════════
# Reconstruction du modèle depuis le checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(cfg_inr, cfg_data, device: torch.device) -> ModulatedFourierFeatures:
    return ModulatedFourierFeatures(
        input_dim        = cfg_data.input_dim,
        output_dim       = cfg_data.output_dim,
        look_back_window = cfg_data.look_back_window,
        num_frequencies  = cfg_inr.num_frequencies,
        latent_dim       = cfg_inr.latent_dim,
        lstm_hidden_dim  = cfg_inr.lstm_hidden_dim,
        spatial_dim      = cfg_inr.static.spatial_dim,
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


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation principale
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
        parquet_path: str,
        batch_size:   int  = 32,
        num_workers:  int  = 0,
        quiet:        bool = False,
) -> dict:
    """
    Charge le checkpoint figé, évalue sur parquet_path.
    Retourne {"mae": float, "rmse": float, "mape": float}.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device  : {device}")
    log(f"Parquet : {parquet_path}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_files = sorted(SNAPSHOT_DIR.glob("*_best.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"Aucun checkpoint (*_best.pt) dans {SNAPSHOT_DIR}")
    ckpt_path = ckpt_files[-1]
    ckpt      = torch.load(ckpt_path, map_location=device)
    log(f"Checkpoint : {ckpt_path.name}  (epoch {ckpt['epoch']}  val={ckpt['val_loss']:.6f})")

    # ── Configs ───────────────────────────────────────────────────────────────
    cfg_inr   = OmegaConf.create(ckpt["cfg_inr"])
    cfg_data  = OmegaConf.create(ckpt["cfg_data"])
    cfg_inner = OmegaConf.create(ckpt["cfg_inner"])
    cfg_optim = OmegaConf.create(ckpt["cfg_optim"])

    # Vérification schéma de données (guard contre mauvais parquet)
    from src.dataloaders import schema_fingerprint, check_compat
    saved_fp = ckpt.get("data_schema_fp", {})
    diffs    = check_compat(saved_fp)
    if diffs:
        log("[ATTENTION] Divergence de schéma détectée :")
        for d in diffs:
            log(f"  {d}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    scalers = joblib.load(SNAPSHOT_DIR / "scalers.pkl")

    # ── Dataset ───────────────────────────────────────────────────────────────
    testset = NZDataset(
        parquet_path,
        mode       = "val",       # val = transform-only, scalers fournis
        latent_dim = cfg_inr.latent_dim,
        scalers    = scalers,
    )
    loader = DataLoader(
        testset,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = False,
    )
    log(f"Fenêtres test : {len(testset)}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = _build_model(cfg_inr, cfg_data, device)
    model.load_state_dict(ckpt["inr_state_dict"])
    model.eval()
    model._debug = False

    # alpha figé (valeur apprise ou fallback sur lr_code)
    alpha_val = ckpt.get("alpha", cfg_optim.lr_code)
    alpha     = torch.tensor([alpha_val], device=device)

    look_back   = cfg_data.look_back_window
    use_context = cfg_inr.use_context
    all_preds, all_targets = [], []

    # ── Inférence ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", disable=quiet)
        for batch in pbar:
            t, dir_idx, x_statics, x_meteo, x_time, y, code = batch

            x_context = x_meteo.to(device)
            y_target  = y.to(device)
            t         = t.to(device)
            code      = code.to(device)

            x_statics = torch.concat(
                [x_statics.unsqueeze(1).repeat(1, x_time.size(1), 1), x_time], dim=-1
            )
            x_statics = x_statics.to(device) if use_context else None
            dir_idx   = dir_idx.to(device)   if use_context else None

            coords_p    = t[:, :look_back, :]
            coords_h    = t[:, look_back:, :]
            x_context_p = x_context[:, :look_back, :]
            x_context_h = x_context[:, look_back:, :]
            y_past      = y_target[:, :look_back, :]
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
                x_statics   = x_statics,
                dir_idx     = dir_idx,
            )

            if "out_h" not in outputs:
                raise RuntimeError(
                    "outer_step ne retourne pas 'out_h'. "
                    "Ajouter la clé dans src/metalearning.py."
                )

            all_preds.append(outputs["out_h"].detach().cpu())
            all_targets.append(y_horizon.detach().cpu())

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds, targets)

    log("\n── Métriques ──────────────────────────")
    for k, v in metrics.items():
        log(f"  {k.upper():>6} : {v:.6f}")
    log("────────────────────────────────────────")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un snapshot figé")
    parser.add_argument("--parquet",     required=True,      help="Chemin vers le .parquet de test")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quiet",       action="store_true", help="Supprime les prints (garde le JSON final)")
    args = parser.parse_args()

    metrics = evaluate(args.parquet, args.batch_size, args.num_workers, args.quiet)

    # Toujours printer le JSON en dernière ligne (parsé par compare_runs.py)
    print("JSON:" + _json.dumps(metrics))