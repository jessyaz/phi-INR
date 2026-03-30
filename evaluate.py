#!/usr/bin/env python
"""
evaluate.py — compare INR snapshots + LSTM baselines, génère rapport HTML.

Principe fondamental :
  Chaque snapshot contient son propre src/ gelé. On charge TOUS les modules
  directement depuis ces fichiers (importlib.util.spec_from_file_location),
  jamais depuis le code source courant. Cela garantit que le modèle, le
  dataloader et les constantes (LSTM_IN_DIM, STAT_DIM…) sont exactement ceux
  utilisés à l'entraînement, même si le code a changé depuis.
"""
from __future__ import annotations

import argparse, base64, importlib.util, io, json, sys, warnings
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52",
           "#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974"]


# ══════════════════════════════════════════════════════════════════════════════
# Chargement direct de fichiers Python (bypass sys.modules / sys.path)
# ══════════════════════════════════════════════════════════════════════════════

def _load_file(unique_name: str, filepath: Path):
    """
    Charge un fichier .py directement, indépendamment de sys.path et sys.modules.
    unique_name doit être distinct pour chaque snapshot afin d'éviter les collisions.
    """
    spec = importlib.util.spec_from_file_location(unique_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_snapshot_modules(snap_dir: Path) -> dict:
    """
    Charge tous les modules src/ du snapshot directement depuis leurs fichiers.
    Retourne un dict { "network": <module>, "metalearning": <module>, ... }
    """
    src = snap_dir / "src"
    n   = snap_dir.name   # préfixe unique par snapshot

    mods = {}
    for name in ["dataloaders", "network", "metalearning",
                 "film_conditionning", "head_sequencer"]:
        f = src / f"{name}.py"
        if f.exists():
            mods[name] = _load_file(f"{n}__{name}", f)
    return mods


# ══════════════════════════════════════════════════════════════════════════════
# Métriques
# ══════════════════════════════════════════════════════════════════════════════

def _m(obs, pred):
    o = obs.flatten().astype(np.float64)
    p = pred.flatten().astype(np.float64)
    m = ~(np.isnan(o) | np.isnan(p) | np.isinf(o) | np.isinf(p))
    return o[m], p[m]

def rmse(o, p):   o, p = _m(o,p); return float(np.sqrt(np.mean((o-p)**2)))
def mae(o, p):    o, p = _m(o,p); return float(np.mean(np.abs(o-p)))

def smape(o, p):
    o, p  = _m(o, p)
    denom = (np.abs(o) + np.abs(p)) / 2.0
    mask  = denom > 1.0
    if mask.sum() == 0: return np.nan
    return float(100 * np.mean(np.abs(o[mask]-p[mask]) / denom[mask]))

def metrics(obs, pred):
    return {"RMSE": rmse(obs,pred), "MAE": mae(obs,pred), "sMAPE": smape(obs,pred)}

def per_smape(obs, pred):
    return np.array([smape(obs[i], pred[i]) for i in range(obs.shape[0])])

def per_mae(obs, pred):
    return np.array([mae(obs[i], pred[i]) for i in range(obs.shape[0])])


# ══════════════════════════════════════════════════════════════════════════════
# Inférence INR snapshot
# ══════════════════════════════════════════════════════════════════════════════

def _build_loader_from_snapshot(snap_dir: Path, mods: dict, test_data_path: Path,
                                scalers, latent_dim: int,
                                batch_size: int, num_workers: int) -> DataLoader:
    """Construit un DataLoader en utilisant le NZDataset gelé du snapshot."""
    NZDataset = mods["dataloaders"].NZDataset
    # Anciens snapshots peuvent ne pas avoir skip_site_filter
    try:
        testset = NZDataset(
            test_data_path, mode="test",
            scalers=scalers, latent_dim=latent_dim,
            skip_site_filter=True,
        )
    except TypeError:
        print("E")
        testset = NZDataset(
            test_data_path, mode="test",
            scalers=scalers, latent_dim=latent_dim,
        )
    print(f"  ✓ Loader gelé : {len(testset)} samples")
    return DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), shuffle=False,
    )


def _patch_frozen_network(mods: dict):
    """
    network.py gelé contient des imports comme `from src.dataloaders import LSTM_IN_DIM`.
    Au moment de l'exécution, Python résout ces imports via sys.modules — ce qui
    donne les constantes du code COURANT, pas celles du snapshot.
    On injecte directement les constantes du dataloader gelé dans le module network gelé.
    """
    dl   = mods.get("dataloaders")
    net  = mods.get("network")
    head = mods.get("head_sequencer")
    if dl is None or net is None:
        return
    # Injecte toutes les constantes exportées par dataloaders gelé dans network gelé
    for attr in ("LSTM_IN_DIM", "STAT_DIM", "X_TIME_DIM", "LOOK_BACK", "HORIZON"):
        if hasattr(dl, attr):
            setattr(net, attr, getattr(dl, attr))
    # head_sequencer a aussi besoin de LSTM_IN_DIM
    if head is not None:
        for attr in ("LSTM_IN_DIM", "STAT_DIM", "X_TIME_DIM"):
            if hasattr(dl, attr):
                setattr(head, attr, getattr(dl, attr))


def _build_inr(snap_dir: Path, mods: dict, ckpt: dict, device: torch.device):
    """Reconstruit ModulatedFourierFeatures depuis les modules gelés du snapshot."""
    # Patch indispensable : injecte les constantes du dataloader gelé dans network gelé
    _patch_frozen_network(mods)

    ci, cd = ckpt["cfg_inr"], ckpt["cfg_data"]
    sd     = ckpt["inr_state_dict"]

    MFF            = mods["network"].ModulatedFourierFeatures
    spatial_dim    = ci["static"]["spatial_dim"]
    dir_dim        = ci["static"]["dir_dim"]
    num_directions = ci["static"]["num_directions"]
    latent_dim     = ci["latent_dim"]

    # AutoFix latent_dim si mismatch
    for key in ("latent_to_mod.net.weight", "latent_to_mod_vanilla.net.weight"):
        if key in sd and sd[key].shape[1] != latent_dim:
            print(f"  [AutoFix] latent_dim {latent_dim} → {sd[key].shape[1]}")
            latent_dim = sd[key].shape[1]
            ci = {**ci, "latent_dim": latent_dim}

    model = MFF(
        input_dim        = cd["input_dim"],
        output_dim       = cd["output_dim"],
        look_back_window = cd["look_back_window"],
        num_frequencies  = ci["num_frequencies"],
        latent_dim       = latent_dim,
        lstm_hidden_dim  = ci["lstm_hidden_dim"],
        spatial_dim      = spatial_dim,
        dir_dim          = dir_dim,
        num_directions   = num_directions,
        sigma            = ci["static"]["sigma"],
        width            = ci["hidden_dim"],
        depth            = ci["depth"],
        min_frequencies  = ci["min_frequencies"],
        base_frequency   = ci["base_frequency"],
        include_input    = ci.get("include_input", True),
        is_training      = False,
        use_context      = ci["use_context"],
        freeze_lstm      = False,
        control          = ci.get("control", None),
    ).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  ⚠ Clés manquantes : {missing[:3]}")
    if unexpected: print(f"  ⚠ Clés ignorées   : {unexpected[:3]}")
    if not missing and not unexpected:
        print(f"  ✓ State dict chargé proprement")

    model.eval()
    model._debug = False
    return model, cd["look_back_window"], ci


def _infer_inr(model, mods: dict, ci: dict, look_back: int,
               full_cfg, loader, device) -> tuple:
    """Inférence INR — utilise outer_step depuis les modules gelés du snapshot."""
    outer_step = mods["metalearning"].outer_step

    if full_cfg:
        inner_steps = int(OmegaConf.select(full_cfg, "inner.inner_steps", default=3))
        inner_lr    = float(OmegaConf.select(full_cfg, "optim.lr_code",   default=1e-2))
        w_p  = float(OmegaConf.select(full_cfg, "inner.w_passed", default=1.0))
        w_f  = float(OmegaConf.select(full_cfg, "inner.w_futur",  default=1.0))
    else:
        inner_steps, inner_lr, w_p, w_f = 3, 1e-2, 1.0, 1.0

    use     = ci["use_context"]
    control = ci.get("control")
    alpha   = torch.tensor([inner_lr], device=device)
    print(f"  use_context={use} | control={control}")

    ph, yh, pp, yp = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  INR", leave=False):
            t, di, xs, xm, xt, y, _ = batch
            yt  = y.to(device); t = t.to(device)
            xc  = xm.to(device)

            cp, ch   = t[:, :look_back, :], t[:, look_back:, :]
            ypa, yho = yt[:, :look_back, :], yt[:, look_back:, :]
            code0    = torch.zeros(t.shape[0], ci["latent_dim"], device=device)

            if use or control == "static_only":
                xt_d = xt.to(device)
                xs_e = xs.to(device).unsqueeze(1).expand(-1, xt_d.size(1), -1)
                xss  = torch.cat([xs_e, xt_d], dim=-1)
                dis  = di.to(device)
            else:
                xss = dis = None

            if use:
                out = outer_step(
                    func_rep    = model,
                    coords_p    = cp, coords_h = ch,
                    x_context_p = xc[:, :look_back, :],
                    x_context_h = xc[:, look_back:, :],
                    y_past      = ypa, y_horizon = yho,
                    inner_steps = inner_steps, inner_lr = alpha,
                    w_passed    = w_p, w_futur = w_f,
                    is_train    = False, code = code0,
                    x_statics   = xss, dir_idx = dis,
                )
                out_p, out_h = out["out_p"], out["out_h"]
            else:
                out_p, out_h, _ = model.modulated_forward(
                    coords_p    = cp, code = code0,
                    x_context_p = xc[:, :look_back, :], y_past = ypa,
                    x_statics   = xss, dir_idx = dis,
                    coords_h    = ch,
                    x_context_h = xc[:, look_back:, :],
                )

            ph.append(out_h.cpu().numpy())
            yh.append(yho.cpu().numpy())
            pp.append(out_p.cpu().numpy())
            yp.append(ypa.cpu().numpy())

    return (np.concatenate(ph, 0).squeeze(-1), np.concatenate(yh, 0).squeeze(-1),
            np.concatenate(pp, 0).squeeze(-1), np.concatenate(yp, 0).squeeze(-1))


def _inverse(arr, scalers):
    if scalers is None: return arr
    out = scalers["target"].inverse_transform(arr.reshape(-1,1)).reshape(arr.shape)
    return np.clip(out, 0, None)


def evaluate_inr_snapshot(snap_dir: Path, test_data_path: Path, device: torch.device,
                          scalers=None, batch_size: int = 64,
                          num_workers: int = 2) -> dict | None:
    """
    Évalue un snapshot INR en utilisant EXCLUSIVEMENT les fichiers gelés dans snap_dir/src/.
    Chaque snapshot construit son propre DataLoader depuis son NZDataset gelé.
    """
    snap_dir = Path(snap_dir)
    ckpts    = sorted(snap_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Pas de .pt dans {snap_dir}")
    if not (snap_dir / "src").is_dir():
        raise FileNotFoundError(f"Pas de src/ gelé dans {snap_dir} — snapshot incomplet")

    print(f"  source : {snap_dir.name}  ❄")

    # ── Chargement de TOUS les modules depuis les fichiers gelés ──────────────
    mods = _load_snapshot_modules(snap_dir)

    cfg_path = snap_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml absent dans {snap_dir}")
    full_cfg  = OmegaConf.load(cfg_path)
    ckpt_path = ckpts[-1]
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model, look_back, ci = _build_inr(snap_dir, mods, ckpt, device)

    # ── Scalers gelés (priorité aux scalers du snapshot) ─────────────────────
    snap_scalers_path = snap_dir / "scalers.pkl"
    snap_scalers = joblib.load(snap_scalers_path) if snap_scalers_path.exists() else scalers
    if not snap_scalers_path.exists():
        print("  ⚠ Pas de scalers gelés → scalers globaux utilisés")

    # ── DataLoader gelé (NZDataset du snapshot) ───────────────────────────────
    snap_loader = _build_loader_from_snapshot(
        snap_dir, mods, test_data_path, snap_scalers,
        latent_dim  = ci["latent_dim"],
        batch_size  = batch_size,
        num_workers = num_workers,
    )

    # ── Inférence ─────────────────────────────────────────────────────────────
    ph, yh, pp, yp = _infer_inr(model, mods, ci, look_back, full_cfg, snap_loader, device)
    ph, yh, pp, yp = (_inverse(x, snap_scalers) for x in (ph, yh, pp, yp))

    return {
        "name":       snap_dir.name,
        "type":       f"INR [{ci.get('control', 'context')}] ❄",
        "pred_h": ph, "y_h": yh, "pred_p": pp, "y_p": yp,
        "val_loss":   ckpt.get("val_loss",   np.nan),
        "val_loss_h": ckpt.get("val_loss_h", np.nan),
        "val_loss_p": ckpt.get("val_loss_p", np.nan),
        "epoch":      ckpt.get("epoch", -1),
        "config":     {"inr": ci, "data": ckpt.get("cfg_data", {})},
        "ckpt_path":  str(ckpt_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Inférence LSTM baseline (code courant — les baselines n'ont pas de snapshot)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_baseline_lstm(ckpt_path: Path, loader, device: torch.device,
                           look_back: int, scalers=None) -> dict | None:
    ckpt_path = Path(ckpt_path)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)

    from src.baseline_lstm import BaselineLSTM, DirEmbedding
    from src.dataloaders   import STAT_DIM, X_TIME_DIM

    sd  = ckpt["model"]
    _ALIAS = {"dynamic_static": "dynamic", "static": "static_only", "flow": "flow_only"}
    raw_mode = ckpt.get("mode", "static_only")
    mode     = _ALIAS.get(raw_mode, raw_mode)
    if mode != raw_mode:
        print(f"  [AutoFix] mode {raw_mode!r} → {mode!r}")

    hidden_dim     = sd["lstm.weight_hh_l0"].shape[1]
    STATIC_SEQ_DIM = STAT_DIM + X_TIME_DIM + 8
    lstm_input     = sd["lstm.weight_ih_l0"].shape[1]
    x_dyn_dim      = max(0, lstm_input - 1 - STATIC_SEQ_DIM) if mode == "dynamic" else 0

    model = BaselineLSTM(hidden_dim=hidden_dim, mode=mode, x_dyn_dim=x_dyn_dim).to(device)
    model.load_state_dict(sd); model.eval()

    dir_emb = DirEmbedding(output_dim=8).to(device)
    if "dir_emb" in ckpt:
        dir_emb.load_state_dict(ckpt["dir_emb"])
    else:
        print("  ⚠ dir_emb absent du checkpoint — poids aléatoires")
    dir_emb.eval()

    print(f"  ✓ LSTM mode={mode!r}  hidden={hidden_dim}  x_dyn_dim={x_dyn_dim}")

    ph, yh, pp, yp = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  LSTM [{mode}]", leave=False):
            _, dir_idx, x_static, x_dyn, x_time, y, _ = batch
            y_d = y.to(device)
            T   = x_time.size(1)

            if mode != "flow_only":
                dv  = dir_emb(dir_idx.to(device))
                xss = torch.cat([
                    x_static.to(device).unsqueeze(1).expand(-1, T, -1),
                    x_time.to(device),
                    dv.unsqueeze(1).expand(-1, T, -1),
                ], dim=-1)
            else:
                xss = None

            preds = model(
                y_flow    = y_d, twin_idx  = look_back,
                x_statics = xss,
                x_dyn     = x_dyn.to(device) if mode == "dynamic" else None,
            )

            B       = preds.shape[0]
            nan_col = np.full((B, 1), np.nan, dtype=np.float32)
            pp.append(np.concatenate([nan_col, preds[:, :look_back-1, :].cpu().numpy().squeeze(-1)], axis=1))
            ph.append(preds[:, look_back-1:, :].cpu().numpy().squeeze(-1))
            yp.append(y[:, :look_back, :].numpy().squeeze(-1))
            yh.append(y[:,  look_back:, :].numpy().squeeze(-1))

    pred_h = _inverse(np.concatenate(ph, 0), scalers)
    y_h    = _inverse(np.concatenate(yh, 0), scalers)
    pred_p = _inverse(np.concatenate(pp, 0), scalers)
    y_p    = _inverse(np.concatenate(yp, 0), scalers)

    return {
        "name":       f"LSTM-{mode}/{ckpt_path.parent.name}",
        "type":       f"LSTM ({mode})",
        "pred_h": pred_h, "y_h": y_h, "pred_p": pred_p, "y_p": y_p,
        "val_loss":   ckpt.get("val_loss",   np.nan),
        "val_loss_h": ckpt.get("val_loss_h", np.nan),
        "val_loss_p": ckpt.get("val_loss_p", np.nan),
        "epoch":      ckpt.get("epoch", -1),
        "config":     {"mode": mode, "hidden_dim": hidden_dim, "x_dyn_dim": x_dyn_dim},
        "ckpt_path":  str(ckpt_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    return f"{v:{fmt}}"

def _rows(results):
    out = []
    for r in results:
        mh = metrics(r["y_h"], r["pred_h"])
        mp = metrics(r["y_p"], r["pred_p"])
        out.append({
            "name":       r["name"],
            "type":       r["type"],
            "epoch":      r["epoch"],
            "val_loss_h": r.get("val_loss_h", np.nan),
            "RMSE_h": mh["RMSE"], "MAE_h": mh["MAE"], "sMAPE_h": mh["sMAPE"],
            "RMSE_p": mp["RMSE"], "MAE_p": mp["MAE"], "sMAPE_p": mp["sMAPE"],
        })
    return out


def fig_bars(rows):
    n      = len(rows)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    names  = [r["name"][:28] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(2.5, .55*n + 1.5)))
    specs = [("MAE_h","MAE — horizon ↓ (véh/h)"),
             ("RMSE_h","RMSE — horizon ↓ (véh/h)"),
             ("sMAPE_h","sMAPE % — horizon ↓")]
    for ax, (key, title) in zip(axes, specs):
        vals = [r[key] for r in rows]
        bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=.4)
        ax.axvline(0, color="#aaa", lw=.8, ls="--")
        ax.set_xlabel(title, fontsize=9)
        fmt    = ".1f" if "sMAPE" in key else ".2f"
        suffix = "%" if "sMAPE" in key else ""
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                xpos = max(v, 0) + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                ax.text(xpos, bar.get_y() + bar.get_height()/2,
                        f"{v:{fmt}}{suffix}", va="center", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.invert_yaxis(); ax.tick_params(labelsize=8)
    fig.suptitle("Métriques comparatives — test set (horizon)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_box(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = [r["name"][:20] for r in results]
    for ax, (fn, title, ylabel) in zip(axes, [
        (per_mae,   "MAE par sample — Horizon",   "MAE (véh/h)"),
        (per_smape, "sMAPE par sample — Horizon", "sMAPE (%)"),
    ]):
        data = [fn(r["y_h"], r["pred_h"]) for r in results]
        data = [d[~np.isnan(d)] for d in data]
        bp   = ax.boxplot(data, labels=labels, patch_artist=True,
                          medianprops={"color":"#111","lw":1.5},
                          flierprops={"marker":".","ms":3,"alpha":.4})
        for patch, c in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(c); patch.set_alpha(.72)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_ts(results, n=6, seed=42):
    rng = np.random.default_rng(seed)

    # Taille minimale commune à tous les résultats (snapshots peuvent avoir
    # des datasets de tailles différentes selon leur schéma de données)
    N_min = min(r["y_h"].shape[0] for r in results)
    T_p   = results[0]["y_p"].shape[1]
    T_h   = results[0]["y_h"].shape[1]

    idxs = rng.choice(N_min, size=min(n, N_min), replace=False)
    sm   = per_smape(results[0]["y_h"][:N_min], results[0]["pred_h"][:N_min])
    sm_i = np.where(np.isnan(sm[idxs]), -np.inf, sm[idxs])
    idxs = idxs[np.argsort(-sm_i)]

    t_all = np.arange(T_p + T_h)
    fig, axes = plt.subplots(len(idxs), 1, figsize=(15, 3.2*len(idxs)), squeeze=False)
    for row, idx in enumerate(idxs):
        ax  = axes[row, 0]
        obs = np.concatenate([results[0]["y_p"][idx], results[0]["y_h"][idx]])
        ax.fill_between(t_all[:T_p], 0, 1, transform=ax.get_xaxis_transform(),
                        alpha=.06, color="#4C72B0")
        ax.plot(t_all, obs, color="#111", lw=1.6, label="Observé", zorder=9)
        ax.axvline(T_p - .5, color="#888", ls=":", lw=.9)
        for i, r in enumerate(results):
            # Vérifie que cet index est valide pour ce résultat
            if idx >= r["y_h"].shape[0]:
                continue
            # Adapte T_p/T_h si ce résultat a des dimensions différentes
            r_T_p = r["y_p"].shape[1]; r_T_h = r["y_h"].shape[1]
            pred  = np.concatenate([r["pred_p"][idx, :r_T_p], r["pred_h"][idx, :r_T_h]])
            t_r   = np.arange(r_T_p + r_T_h)
            tv    = t_r[~np.isnan(pred)]; pv = pred[~np.isnan(pred)]
            sm_v  = smape(r["y_h"][idx], r["pred_h"][idx])
            mae_v = mae(r["y_h"][idx],   r["pred_h"][idx])
            ax.plot(tv, pv, color=PALETTE[i % len(PALETTE)], lw=1.1, alpha=.88,
                    label=f"{r['name'][:16]} sMAPE={sm_v:.1f}% MAE={mae_v:.1f}")
        ax.set_title(f"Sample #{idx}", fontsize=8)
        ax.set_ylabel("Flux (véh/h)", fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=7, ncol=min(len(results)+1, 4),
                      loc="upper left", framealpha=.7)
    axes[-1, 0].set_xlabel("Pas de temps (h)", fontsize=8)
    fig.suptitle("Séries temporelles — pires sMAPE horizon en premier | zone bleue = passé",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_scatter(results):
    n = len(results); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows), squeeze=False)
    for i, r in enumerate(results):
        ax  = axes[i//cols][i%cols]
        obs = r["y_h"].flatten(); pred = r["pred_h"].flatten()
        m   = ~(np.isnan(obs)|np.isnan(pred)|np.isinf(obs)|np.isinf(pred))
        ax.scatter(obs[m], pred[m], alpha=.12, s=3,
                   color=PALETTE[i % len(PALETTE)], rasterized=True)
        lim = [min(obs[m].min(), pred[m].min()), max(obs[m].max(), pred[m].max())]
        ax.plot(lim, lim, "k--", lw=.9)
        ax.set_title(f"{r['name'][:24]}\nsMAPE={smape(obs[m],pred[m]):.1f}%  "
                     f"RMSE={rmse(obs[m],pred[m]):.1f}", fontsize=8)
        ax.set_xlabel("Observé (véh/h)", fontsize=8)
        ax.set_ylabel("Prédit (véh/h)",  fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
    for j in range(i+1, rows*cols):
        axes[j//cols][j%cols].set_visible(False)
    fig.suptitle("Scatter Observé vs Prédit — Horizon", fontsize=10, fontweight="bold")
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_errors(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    ax = axes[0]
    for i, r in enumerate(results):
        errs = np.abs(r["y_h"].flatten() - r["pred_h"].flatten())
        ax.hist(errs[~np.isnan(errs)], bins=80, alpha=.5,
                color=PALETTE[i % len(PALETTE)], label=r["name"][:20], density=True)
    ax.set_xlabel("Erreur absolue (véh/h)", fontsize=8); ax.set_ylabel("Densité", fontsize=8)
    ax.set_title("Distribution MAE — Horizon", fontsize=9)
    ax.legend(fontsize=7, framealpha=.7); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax = axes[1]
    for i, r in enumerate(results):
        o, p  = _m(r["y_h"].flatten(), r["pred_h"].flatten())
        denom = (np.abs(o) + np.abs(p)) / 2.0; mask = denom > 1.0
        if mask.sum() > 0:
            ax.hist(np.clip(100*np.abs(o[mask]-p[mask])/denom[mask], 0, 200),
                    bins=80, alpha=.5, color=PALETTE[i % len(PALETTE)],
                    label=r["name"][:20], density=True)
    ax.set_xlabel("sMAPE (%) — tronquée à 200%", fontsize=8)
    ax.set_ylabel("Densité", fontsize=8); ax.set_title("Distribution sMAPE — Horizon", fontsize=9)
    ax.legend(fontsize=7, framealpha=.7); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
:root{--a:#4C72B0;--bg:#f7f8fa}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:#1a1a2e;font-size:14px;line-height:1.6;padding:32px 52px}
h1{font-size:1.7rem;color:var(--a);margin-bottom:4px}
h2{font-size:1.1rem;margin:36px 0 10px;padding-bottom:5px;
   border-bottom:2px solid var(--a);color:#333;font-weight:600}
.meta{color:#6c757d;font-size:.83rem;margin-bottom:28px}
table{border-collapse:collapse;width:100%;font-size:.82rem;background:#fff;
      border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)}
th{background:var(--a);color:#fff;padding:9px 12px;text-align:left;font-weight:500}
td{padding:7px 12px;border-bottom:1px solid #eee}
tr:nth-child(even) td{background:#f8f9fb}
tr:last-child td{border-bottom:none}
.best{font-weight:700;color:#198754}
.fig{margin:18px 0}.fig img{max-width:100%;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.1)}
.badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:.75rem;font-weight:600}
.bi{background:#cfe2ff;color:#084298}.bc{background:#d4edda;color:#155724}
.bl{background:#fff3cd;color:#664d03}.bo{background:#e9ecef;color:#495057}
.lstm-flow{background:#fce4ec;color:#880e4f}
.lstm-static{background:#e8f5e9;color:#1b5e20}
.lstm-dynamic{background:#e3f2fd;color:#0d47a1}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
details{margin:6px 0}details summary{cursor:pointer;font-weight:500;padding:4px 0;color:#444}
pre{background:#f3f4f6;padding:12px 16px;border-radius:6px;font-size:.77rem;overflow-x:auto;margin-top:6px}
footer{margin-top:48px;font-size:.75rem;color:#adb5bd}
@media(max-width:900px){body{padding:16px 18px}.grid2{grid-template-columns:1fr}}
"""

def _safe_json(obj):
    if isinstance(obj, dict):   return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe_json(v) for v in obj]
    if hasattr(obj, "item"):    return obj.item()
    if hasattr(obj, "tolist"):  return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None: return obj
    return str(obj)

def _table(rows):
    best = {}
    for col in ("RMSE_h","MAE_h","sMAPE_h","RMSE_p","MAE_p","sMAPE_p","val_loss_h"):
        vals = [r[col] for r in rows if not np.isnan(r[col])]
        if vals: best[col] = min(vals)

    def isbest(col, val):
        b = best.get(col)
        return b is not None and not np.isnan(val) and abs(val - b) < 1e-6

    def td(col, val, fmt=".2f"):
        c = ' class="best"' if isbest(col, val) else ""
        return f"<td{c}>{_fmt(val, fmt)}</td>"

    def td_pct(col, val):
        c = ' class="best"' if isbest(col, val) else ""
        return f"<td{c}>{_fmt(val, '.1f')}%</td>"

    def _badge(t):
        lut = {
            "INR [static_only] ❄": ("bi",          "INR static ❄"),
            "INR [static_only]":   ("bi",           "INR static"),
            "INR [context] ❄":     ("bc",           "INR LSTM ❄"),
            "INR [dynamic] ❄":     ("bc",           "INR dynamic ❄"),
            "INR [None] ❄":        ("bc",           "INR ❄"),
            "LSTM (flow_only)":    ("lstm-flow",    "LSTM flow"),
            "LSTM (static_only)":  ("lstm-static",  "LSTM static"),
            "LSTM (dynamic)":      ("lstm-dynamic", "LSTM dynamic"),
        }
        bc, bt = lut.get(t, ("bo", t[:20]))
        return f'<span class="badge {bc}">{bt}</span>'

    hdr = """<table><thead><tr>
  <th>Nom</th><th>Type</th><th>Epoch</th><th>val_loss_h ↓</th>
  <th>RMSE_h ↓</th><th>MAE_h ↓</th><th>sMAPE_h ↓</th>
  <th>RMSE_p ↓</th><th>MAE_p ↓</th><th>sMAPE_p ↓</th>
</tr></thead><tbody>"""
    body = ""
    for r in rows:
        body += f"""<tr>
  <td title="{r['name']}">{r['name'][:34]}</td>
  <td>{_badge(r['type'])}</td><td>{r['epoch']}</td>
  {td('val_loss_h', r['val_loss_h'], '.4f')}
  {td('RMSE_h',r['RMSE_h'])}{td('MAE_h',r['MAE_h'])}{td_pct('sMAPE_h',r['sMAPE_h'])}
  {td('RMSE_p',r['RMSE_p'])}{td('MAE_p',r['MAE_p'])}{td_pct('sMAPE_p',r['sMAPE_p'])}
</tr>"""
    return hdr + body + "</tbody></table>"


_HTML = """<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title><style>{css}</style></head><body>
<h1>{title}</h1>
<div class="meta">Généré le {date} · {nm} modèle(s) · look_back={lb}h · horizon={hz}h</div>
<h2>Métriques globales — test set</h2>{table}
<h2>MAE / RMSE / sMAPE comparatifs (horizon)</h2>
<div class="fig"><img src="data:image/png;base64,{b_bars}"/></div>
<h2>Distribution MAE et sMAPE par sample</h2>
<div class="fig"><img src="data:image/png;base64,{b_box}"/></div>
<h2>Séries temporelles (pires sMAPE horizon en premier)</h2>
<div class="fig"><img src="data:image/png;base64,{b_ts}"/></div>
<div class="grid2">
  <div><h2>Scatter Observé vs Prédit (horizon)</h2>
    <div class="fig"><img src="data:image/png;base64,{b_sc}"/></div></div>
  <div><h2>Distribution MAE et sMAPE</h2>
    <div class="fig"><img src="data:image/png;base64,{b_err}"/></div></div>
</div>
<h2>Configurations</h2>{configs}
<footer>evaluate.py · {date}</footer></body></html>"""


def generate(results, output_dir, look_back, horizon, title="Comparaison"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(results)
    print("  figures …")
    b_bars = fig_bars(rows);  b_box = fig_box(results)
    b_ts   = fig_ts(results); b_sc  = fig_scatter(results); b_err = fig_errors(results)

    configs = ""
    for r in results:
        cfg_str = json.dumps(_safe_json(r.get("config", {})), indent=2, ensure_ascii=False)
        configs += (f"<details><summary>{r['name']}</summary>"
                    f"<p style='font-size:.8rem;color:#888'>Checkpoint : {r.get('ckpt_path','—')}</p>"
                    f"<pre>{cfg_str}</pre></details>")

    html = _HTML.format(
        title=title, css=_CSS,
        date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        nm=len(results), lb=look_back, hz=horizon,
        table=_table(rows),
        b_bars=b_bars, b_box=b_box, b_ts=b_ts, b_sc=b_sc, b_err=b_err,
        configs=configs,
    )
    out = output_dir / "report.html"
    out.write_text(html, encoding="utf-8")

    metrics_out = [
        {"name": r["name"], "type": r["type"],
         **{k: (None if np.isnan(float(v)) else float(v))
            for k, v in row.items() if isinstance(v, (int, float))}}
        for r, row in zip(results, rows)
    ]
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False)
    )
    print(f"\n✓  {out}\n   {output_dir / 'metrics.json'}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python evaluate.py \\
    --snapshots save_models/snapshots/AugC-INR/gifted-colt-894 \\
                save_models/snapshots/AugC-INR/gregarious-conch-145 \\
    --baselines save_models/baseline_lstm_dynamic/run_best.pt \\
    --test_data data/test_data.parquet
""")
    p.add_argument("--snapshots", "--ckpts", nargs="*", default=[], metavar="DIR",
                   help="Dossiers snapshot INR (doivent contenir src/ et scalers.pkl)")
    p.add_argument("--baselines",   nargs="*", default=[], metavar="FILE",
                   help="Checkpoints BaselineLSTM .pt")
    p.add_argument("--test_data",   default="data/test_data.parquet")
    p.add_argument("--scalers",     default=None,
                   help="Scalers globaux pour les baselines (fallback)")
    p.add_argument("--look_back",   type=int, default=None)
    p.add_argument("--horizon",     type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output",      default="results/comparison")
    p.add_argument("--title",       default="Comparaison INR / LSTM")
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    if not any([args.snapshots, args.baselines]):
        p.error("Spécifie --snapshots et/ou --baselines")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")

    test_data_path = ROOT / args.test_data

    # ── Scalers globaux (pour les baselines LSTM) ─────────────────────────────
    global_scalers = None
    if args.scalers and Path(args.scalers).exists():
        global_scalers = joblib.load(args.scalers)
        print(f"Scalers globaux : {args.scalers}")
    else:
        print("⚠ Pas de scalers globaux — chaque snapshot utilisera ses propres scalers.pkl")

    # ── look_back / horizon depuis le premier snapshot ────────────────────────
    lb, hz = args.look_back, args.horizon
    if lb is None and args.snapshots:
        try:
            cfg = OmegaConf.load(Path(args.snapshots[0]) / "config.yaml")
            lb  = int(OmegaConf.select(cfg, "data.look_back_window", default=192))
            hz  = int(OmegaConf.select(cfg, "data.horizon",          default=48))
        except Exception as e:
            print(f"  ⚠ Impossible de lire lb/hz : {e}")
    lb = lb or 192; hz = hz or 48
    print(f"look_back={lb}  horizon={hz}")

    results = []

    # ── INR snapshots — chacun avec son propre code + loader gelés ───────────
    for s in args.snapshots:
        print(f"\n→ INR snapshot : {s}")
        try:
            r = evaluate_inr_snapshot(
                snap_dir       = Path(s),
                test_data_path = test_data_path,
                device         = device,
                scalers        = global_scalers,
                batch_size     = args.batch_size,
                num_workers    = args.num_workers,
            )
            if r:
                results.append(r)
                m = metrics(r["y_h"], r["pred_h"])
                print(f"   MAE_h={m['MAE']:.2f}  RMSE_h={m['RMSE']:.2f}  sMAPE_h={m['sMAPE']:.1f}%")
        except Exception as e:
            print(f"   ✗ {e}"); import traceback; traceback.print_exc()

    # ── Baselines LSTM — loader partagé avec code courant ────────────────────
    if args.baselines:
        from src.dataloaders import NZDataset
        testset = NZDataset(
            test_data_path, mode="test",
            scalers=global_scalers, latent_dim=256,
            skip_site_filter=True,
        )
        baseline_loader = DataLoader(
            testset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"), shuffle=False,
        )
        print(f"\nSamples test (baselines) : {len(testset)}")

        for bl in args.baselines:
            print(f"\n→ LSTM baseline : {bl}")
            try:
                r = evaluate_baseline_lstm(Path(bl), baseline_loader, device, lb, global_scalers)
                if r:
                    results.append(r)
                    m = metrics(r["y_h"], r["pred_h"])
                    print(f"   {r['type']}  MAE_h={m['MAE']:.2f}  "
                          f"RMSE_h={m['RMSE']:.2f}  sMAPE_h={m['sMAPE']:.1f}%")
            except Exception as e:
                print(f"   ✗ {e}"); import traceback; traceback.print_exc()

    if not results:
        print("\n✗ Aucun modèle évalué."); return

    print(f"\nGénération du rapport → {args.output}/")
    generate(results, Path(args.output), lb, hz, args.title)


if __name__ == "__main__":
    main()