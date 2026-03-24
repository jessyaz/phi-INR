#!/usr/bin/env python
"""
evaluate.py — compare INR runs + LSTM baseline, génère rapport HTML
"""
from __future__ import annotations
import argparse, base64, io, json, sys, warnings
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
# Métriques
# ══════════════════════════════════════════════════════════════════════════════

def _m(obs, pred):
    o = obs.flatten().astype(np.float64)
    p = pred.flatten().astype(np.float64)
    m = ~(np.isnan(o)|np.isnan(p)|np.isinf(o)|np.isinf(p))
    return o[m], p[m]

def rmse(o, p):
    o, p = _m(o, p)
    return float(np.sqrt(np.mean((o - p) ** 2)))

def mae(o, p):
    o, p = _m(o, p)
    return float(np.mean(np.abs(o - p)))

def smape(o, p):
    """Symmetric MAPE — bornée [0, 200%], robuste aux flux quasi-nuls.
    Exclut les pas où obs ET pred sont quasi-nuls (dénominateur < 1 veh/h).
    """
    o, p  = _m(o, p)
    denom = (np.abs(o) + np.abs(p)) / 2.0
    mask  = denom > 1.0
    if mask.sum() == 0:
        return np.nan
    return float(100 * np.mean(np.abs(o[mask] - p[mask]) / denom[mask]))

def metrics(obs, pred):
    return {
        "RMSE":  rmse(obs, pred),
        "MAE":   mae(obs, pred),
        "sMAPE": smape(obs, pred),
    }

def per_smape(obs, pred):
    return np.array([smape(obs[i], pred[i]) for i in range(obs.shape[0])])

def per_mae(obs, pred):
    return np.array([mae(obs[i], pred[i]) for i in range(obs.shape[0])])


# ══════════════════════════════════════════════════════════════════════════════
# Inférence INR
# ══════════════════════════════════════════════════════════════════════════════

def _detect_encoder_type(state_dict: dict) -> str:
    keys = set(state_dict.keys())
    if any("static_encoder.spatial_enc" in k for k in keys):
        return "LAST"
    return "MLP"


def _build_model(ckpt: dict, device: torch.device, frozen: bool = False):
    ci, cd = ckpt["cfg_inr"], ckpt["cfg_data"]
    sd     = ckpt["inr_state_dict"]
    encoder_type = _detect_encoder_type(sd)

    from src.network import ModulatedFourierFeatures
    try:
        from src.network import StaticEncoder_LAST, StaticEncoder
        has_both = True
    except ImportError:
        has_both = False

    spatial_dim    = ci["static"]["spatial_dim"]
    dir_dim        = ci["static"]["dir_dim"]
    num_directions = ci["static"]["num_directions"]
    latent_dim     = ci["latent_dim"]

    if "latent_to_mod.net.weight" in sd:
        sd_latent = sd["latent_to_mod.net.weight"].shape[1]
    if "latent_to_mod_vanilla.net.weight" in sd:
        sd_latent = sd["latent_to_mod_vanilla.net.weight"].shape[1]
        if sd_latent != latent_dim:
            print(f"  [AutoFix] latent_dim cfg={latent_dim} → sd={sd_latent}")
            latent_dim = sd_latent
            ci = {**ci, "latent_dim": latent_dim}

    m = ModulatedFourierFeatures(
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
        include_input    = ci["include_input"],
        is_training      = False,
        use_context      = ci["use_context"],
        freeze_lstm      = False,
        control          = ci.get("control", None),
    ).to(device)

    if has_both and encoder_type == "LAST":
        if not hasattr(m.static_encoder, 'spatial_enc'):
            print(f"  [AutoFix] StaticEncoder → StaticEncoder_LAST")
            enc = StaticEncoder_LAST(
                spatial_dim=spatial_dim, dir_dim=dir_dim,
                num_directions=num_directions, sigma=ci["static"]["sigma"],
            ).to(device)
            if any("static_encoder.norm" in k for k in sd):
                enc.norm = nn.LayerNorm(enc.out_dim).to(device)
            m.static_encoder = enc

    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:    print(f"  ⚠ Clés manquantes ({len(missing)}) : {missing[:5]}")
    if unexpected: print(f"  ⚠ Clés ignorées   ({len(unexpected)}) : {unexpected[:5]}")
    if not missing and not unexpected: print(f"  ✓ State dict chargé proprement")
    elif not missing: print(f"  ✓ State dict chargé (clés inattendues ignorées)")

    m.eval(); m._debug = False
    return m, cd["look_back_window"], ci


def _infer_inr(model, ci, look_back, full_cfg, loader, device):
    if full_cfg:
        inner_steps = int(OmegaConf.select(full_cfg, "inner.inner_steps", default=3))
        inner_lr    = float(OmegaConf.select(full_cfg, "optim.lr_code",   default=1e-2))
        w_p = float(OmegaConf.select(full_cfg, "inner.w_passed", default=1.0))
        w_f = float(OmegaConf.select(full_cfg, "inner.w_futur",  default=1.0))
    else:
        inner_steps, inner_lr, w_p, w_f = 3, 1e-2, 1.0, 1.0

    from src.metalearning import outer_step
    alpha = torch.tensor([inner_lr], device=device)
    ph, yh, pp, yp = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  INR", leave=False):
            t, di, xs, xm, xt, y, code = batch
            xc  = torch.cat([xm, xt], dim=-1).to(device)
            yt  = y.to(device); t = t.to(device); code = code.to(device)
            use = ci["use_context"]
            xss = xs.to(device) if use else None
            dis = di.to(device) if use else None
            cp, ch   = t[:, :look_back, :], t[:, look_back:, :]
            ypa, yho = yt[:, :look_back, :], yt[:, look_back:, :]

            out = outer_step(
                func_rep    = model,
                coords_p    = cp, coords_h = ch,
                x_context_p = xc[:, :look_back, :],
                x_context_h = xc[:, look_back:, :],
                y_past      = ypa, y_horizon = yho,
                inner_steps = inner_steps, inner_lr = alpha,
                w_passed    = w_p, w_futur = w_f,
                is_train    = False,
                code        = torch.zeros(t.shape[0], ci["latent_dim"], device=device),
                x_statics   = xss, dir_idx = dis,
            )
            ph.append(out["out_h"].cpu().numpy())
            yh.append(yho.cpu().numpy())
            pp.append(out["out_p"].cpu().numpy())
            yp.append(ypa.cpu().numpy())

    return (np.concatenate(ph, 0).squeeze(-1), np.concatenate(yh, 0).squeeze(-1),
            np.concatenate(pp, 0).squeeze(-1), np.concatenate(yp, 0).squeeze(-1))


def _inverse(arr, scalers):
    if scalers is None: return arr
    sc  = scalers["target"]
    out = sc.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
    return np.clip(out, 0, None)   # contrainte physique : flux ≥ 0


def _compat_check(snap_dir: Path) -> bool:
    from src.snapshot import check_schema_compat
    ok, msgs = check_schema_compat(snap_dir)
    if msgs and not ok:
        print(f"\n{'━'*60}")
        for m in msgs: print(f"  {m}")
        print(f"{'━'*60}")
    elif msgs:
        for m in msgs: print(f"  {m}")
    return ok


def evaluate_inr_ckpt(ckpt_path: Path, loader, device, scalers=None) -> dict | None:
    ckpt_path = Path(ckpt_path)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_path  = ckpt_path.parent / "config.yaml"
    full_cfg  = OmegaConf.load(cfg_path) if cfg_path.exists() else None

    if (ckpt_path.parent / "data_schema_fingerprint.json").exists():
        _compat_check(ckpt_path.parent)

    print(f"  source : {ckpt_path.name}  (code actuel)")
    model, lb, ci = _build_model(ckpt, device)
    ph, yh, pp, yp = _infer_inr(model, ci, lb, full_cfg, loader, device)
    ph, yh, pp, yp = (_inverse(x, scalers) for x in (ph, yh, pp, yp))

    return {"name": f"{ckpt_path.parent.name}/{ckpt_path.stem.replace('_best','')}",
            "type": f"INR [{ci.get('control','context')}]",
            "pred_h": ph, "y_h": yh, "pred_p": pp, "y_p": yp,
            "val_loss":   ckpt.get("val_loss",   np.nan),
            "val_loss_h": ckpt.get("val_loss_h", np.nan),
            "val_loss_p": ckpt.get("val_loss_p", np.nan),
            "epoch":     ckpt.get("epoch", -1),
            "config":    {"inr": ci, "data": ckpt.get("cfg_data", {})},
            "ckpt_path": str(ckpt_path)}


def evaluate_inr_snapshot(snap_dir: Path, loader, device, scalers=None,
                          skip_compat: bool = False) -> dict | None:
    snap_dir = Path(snap_dir)
    ckpts    = sorted(snap_dir.glob("*.pt"))
    if not ckpts: raise FileNotFoundError(f"Pas de .pt dans {snap_dir}")

    if not skip_compat:
        ok = _compat_check(snap_dir)
        if not ok:
            print(f"  ✗ Run ignoré (schéma incompatible) : {snap_dir.name}")
            return None

    from src.snapshot import frozen_src_context
    ckpt_path = ckpts[-1]
    full_cfg  = OmegaConf.load(snap_dir / "config.yaml")
    print(f"  source : {snap_dir.name}  (code gelé ❄)")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    with frozen_src_context(snap_dir):
        model, lb, ci = _build_model(ckpt, device, frozen=True)
        ph, yh, pp, yp = _infer_inr(model, ci, lb, full_cfg, loader, device)

    ph, yh, pp, yp = (_inverse(x, scalers) for x in (ph, yh, pp, yp))

    return {"name": snap_dir.name,
            "type": f"INR [{ci.get('control','context')}] ❄",
            "pred_h": ph, "y_h": yh, "pred_p": pp, "y_p": yp,
            "val_loss":   ckpt.get("val_loss",   np.nan),
            "val_loss_h": ckpt.get("val_loss_h", np.nan),
            "val_loss_p": ckpt.get("val_loss_p", np.nan),
            "epoch":     ckpt.get("epoch", -1),
            "config":    {"inr": ci, "data": ckpt.get("cfg_data", {})},
            "ckpt_path": str(ckpt_path)}


def evaluate_baseline_lstm(ckpt_path: Path, loader, device, look_back,
                           scalers=None, mode="dynamic_static") -> dict | None:
    ckpt_path = Path(ckpt_path)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode      = ckpt.get("mode", mode)

    from src.baseline_lstm import BaselineLSTM
    sd = ckpt["model"]
    p  = {}
    if "lstm.weight_hh_l0" in sd:
        p["hidden_dim"] = sd["lstm.weight_hh_l0"].shape[1]
    if "static_encoder.dir_embedding.weight" in sd:
        p["num_directions"] = sd["static_encoder.dir_embedding.weight"].shape[0]
        p["dir_dim"]        = sd["static_encoder.dir_embedding.weight"].shape[1]
    if "static_encoder.mlp.2.weight" in sd:
        p["spatial_dim"] = sd["static_encoder.mlp.2.weight"].shape[0]

    model = BaselineLSTM(
        hidden_dim     = p.get("hidden_dim", 128),
        mode           = mode,
        spatial_dim    = p.get("spatial_dim", 32),
        dir_dim        = p.get("dir_dim", 8),
        num_directions = p.get("num_directions", 7),
    ).to(device)
    model.load_state_dict(sd); model.eval()
    needs = mode in ("dynamic_static", "static_only")

    ph, yh, pp, yp = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  LSTM [{mode}]", leave=False):
            _, di, xs, xm, xt, y, _ = batch
            xd = torch.cat([xm, xt], dim=-1).to(device)
            yd = y.to(device)
            xs = xs.to(device)
            di = di.to(device)

            preds = model(xd, yd, look_back,
                          x_statics = xs if needs else None,
                          dir_idx   = di if needs else None)
            # preds : (B, T-1, 1)
            # t=0 non prédit par construction → NaN explicite, exclu des métriques
            B       = preds.shape[0]
            nan_col = np.full((B, 1), np.nan, dtype=np.float32)

            pred_p_np = preds[:, :look_back-1, :].cpu().numpy().squeeze(-1)
            pred_h_np = preds[:, look_back-1:, :].cpu().numpy().squeeze(-1)

            pp.append(np.concatenate([nan_col, pred_p_np], axis=1))  # (B, look_back)
            ph.append(pred_h_np)                                      # (B, horizon)
            yp.append(y[:, :look_back, :].numpy().squeeze(-1))
            yh.append(y[:, look_back:,  :].numpy().squeeze(-1))

    pred_h, y_h = np.concatenate(ph, 0), np.concatenate(yh, 0)
    pred_p, y_p = np.concatenate(pp, 0), np.concatenate(yp, 0)
    pred_h, y_h = _inverse(pred_h, scalers), _inverse(y_h, scalers)
    pred_p, y_p = _inverse(pred_p, scalers), _inverse(y_p, scalers)

    return {"name":     f"LSTM-{mode}/{ckpt_path.parent.name}",
            "type":     f"LSTM ({mode})",
            "pred_h": pred_h, "y_h": y_h, "pred_p": pred_p, "y_p": y_p,
            "val_loss":   ckpt.get("val_loss",   np.nan),
            "val_loss_h": ckpt.get("val_loss_h", np.nan),
            "val_loss_p": ckpt.get("val_loss_p", np.nan),
            "epoch":     ckpt.get("epoch", -1),
            "config":    {"mode": mode, "params": p},
            "ckpt_path": str(ckpt_path)}


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
            "RMSE_h":  mh["RMSE"], "MAE_h":  mh["MAE"], "sMAPE_h": mh["sMAPE"],
            "RMSE_p":  mp["RMSE"], "MAE_p":  mp["MAE"], "sMAPE_p": mp["sMAPE"],
        })
    return out


def fig_bars(rows):
    n      = len(rows)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    names  = [r["name"][:28] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(2.5, .55*n + 1.5)))
    specs = [
        ("MAE_h",   "MAE — horizon ↓ (véh/h)"),
        ("RMSE_h",  "RMSE — horizon ↓ (véh/h)"),
        ("sMAPE_h", "sMAPE % — horizon ↓"),
    ]

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
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)

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
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_ts(results, n=6, seed=42):
    rng  = np.random.default_rng(seed)
    N    = results[0]["y_h"].shape[0]
    T_p  = results[0]["y_p"].shape[1]
    T_h  = results[0]["y_h"].shape[1]
    idxs = rng.choice(N, size=min(n, N), replace=False)

    sm    = per_smape(results[0]["y_h"], results[0]["pred_h"])
    sm_i  = np.where(np.isnan(sm[idxs]), -np.inf, sm[idxs])
    idxs  = idxs[np.argsort(-sm_i)]

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
            pred       = np.concatenate([r["pred_p"][idx], r["pred_h"][idx]])
            t_valid    = t_all[~np.isnan(pred)]
            pred_valid = pred[~np.isnan(pred)]
            sm_v  = smape(r["y_h"][idx], r["pred_h"][idx])
            mae_v = mae(r["y_h"][idx],   r["pred_h"][idx])
            lbl   = f"{r['name'][:16]} sMAPE={sm_v:.1f}% MAE={mae_v:.1f}"
            ax.plot(t_valid, pred_valid, color=PALETTE[i % len(PALETTE)],
                    lw=1.1, alpha=.88, label=lbl)

        ax.set_title(f"Sample #{idx}", fontsize=8)
        ax.set_ylabel("Flux (véh/h)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=7, ncol=min(len(results)+1, 4),
                      loc="upper left", framealpha=.7)

    axes[-1, 0].set_xlabel("Pas de temps (h)", fontsize=8)
    fig.suptitle("Séries temporelles | pires sMAPE horizon en premier | zone bleue = passé",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    b = _b64(fig); plt.close(fig); return b


def fig_scatter(results):
    n = len(results); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows), squeeze=False)
    for i, r in enumerate(results):
        ax   = axes[i//cols][i%cols]
        obs  = r["y_h"].flatten(); pred = r["pred_h"].flatten()
        m    = ~(np.isnan(obs)|np.isnan(pred)|np.isinf(obs)|np.isinf(pred))
        ax.scatter(obs[m], pred[m], alpha=.12, s=3,
                   color=PALETTE[i % len(PALETTE)], rasterized=True)
        lim = [min(obs[m].min(), pred[m].min()), max(obs[m].max(), pred[m].max())]
        ax.plot(lim, lim, "k--", lw=.9)
        sm_v   = smape(obs[m], pred[m])
        rmse_v = rmse(obs[m], pred[m])
        ax.set_title(f"{r['name'][:24]}\nsMAPE={sm_v:.1f}%  RMSE={rmse_v:.1f}", fontsize=8)
        ax.set_xlabel("Observé (véh/h)", fontsize=8)
        ax.set_ylabel("Prédit (véh/h)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
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
    ax.set_xlabel("Erreur absolue (véh/h)", fontsize=8)
    ax.set_ylabel("Densité", fontsize=8)
    ax.set_title("Distribution MAE — Horizon", fontsize=9)
    ax.legend(fontsize=7, framealpha=.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    for i, r in enumerate(results):
        o, p  = _m(r["y_h"].flatten(), r["pred_h"].flatten())
        denom = (np.abs(o) + np.abs(p)) / 2.0
        mask  = denom > 1.0
        if mask.sum() > 0:
            sm_errs = 100 * np.abs(o[mask] - p[mask]) / denom[mask]
            ax.hist(np.clip(sm_errs, 0, 200), bins=80, alpha=.5,
                    color=PALETTE[i % len(PALETTE)], label=r["name"][:20], density=True)
    ax.set_xlabel("sMAPE (%) — tronquée à 200%", fontsize=8)
    ax.set_ylabel("Densité", fontsize=8)
    ax.set_title("Distribution sMAPE — Horizon", fontsize=9)
    ax.legend(fontsize=7, framealpha=.7)
    ax.spines[["top","right"]].set_visible(False)
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
.fig{margin:18px 0}
.fig img{max-width:100%;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.1)}
.badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:.75rem;font-weight:600}
.bi{background:#cfe2ff;color:#084298}.bc{background:#d4edda;color:#155724}
.bl{background:#fff3cd;color:#664d03}.bo{background:#e9ecef;color:#495057}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
details{margin:6px 0}details summary{cursor:pointer;font-weight:500;padding:4px 0;color:#444}
pre{background:#f3f4f6;padding:12px 16px;border-radius:6px;
    font-size:.77rem;overflow-x:auto;margin-top:6px}
.schema-warn{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;
             padding:10px 14px;margin:12px 0;font-size:.83rem;color:#664d03}
footer{margin-top:48px;font-size:.75rem;color:#adb5bd}
@media(max-width:900px){body{padding:16px 18px}.grid2{grid-template-columns:1fr}}
"""

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

    badges = {
        "INR [static_only]":   ("bi", "INR static"),
        "INR [static_only] ❄": ("bi", "INR static ❄"),
        "INR [context]":       ("bc", "INR LSTM"),
        "INR [None]":          ("bc", "INR"),
    }

    hdr = """<table><thead><tr>
  <th>Nom</th><th>Type</th><th>Epoch</th><th>val_loss_h ↓</th>
  <th>RMSE_h ↓</th><th>MAE_h ↓</th><th>sMAPE_h ↓</th>
  <th>RMSE_p ↓</th><th>MAE_p ↓</th><th>sMAPE_p ↓</th>
</tr></thead><tbody>"""
    body = ""
    for r in rows:
        bc, bt = badges.get(r["type"], ("bo", r["type"][:18]))
        body += f"""<tr>
  <td title="{r['name']}">{r['name'][:34]}</td>
  <td><span class="badge {bc}">{bt}</span></td>
  <td>{r['epoch']}</td>
  {td('val_loss_h', r['val_loss_h'], '.4f')}
  {td('RMSE_h',  r['RMSE_h'])}{td('MAE_h',  r['MAE_h'])}{td_pct('sMAPE_h', r['sMAPE_h'])}
  {td('RMSE_p',  r['RMSE_p'])}{td('MAE_p',  r['MAE_p'])}{td_pct('sMAPE_p', r['sMAPE_p'])}
</tr>"""
    return hdr + body + "</tbody></table>"


_HTML = """<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title><style>{css}</style></head><body>
<h1>{title}</h1>
<div class="meta">Généré le {date} · {nm} modèle(s) · {ns} samples test · look_back={lb}h · horizon={hz}h</div>
{schema_warn}
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
<h2>Protocole</h2>
<p style="font-size:.88rem;color:#555">
Métriques sur le test set après dénormalisation (véhicules/heure). Flux clippé à 0 (contrainte physique).<br/>
<b>Passé (P)</b> : {lb} premières heures — contexte observé disponible.<br/>
<b>Horizon (H)</b> : {hz} heures suivantes — prédiction auto-régressive.<br/>
<b>Note LSTM</b> : t=0 signalé NaN (pas de prédiction possible sans historique), exclu des métriques.<br/>
<b>sMAPE</b> = 100 × |obs−pred| / ((|obs|+|pred|)/2) — symétrique, bornée [0, 200%],
robuste aux flux quasi-nuls (exclus si dénominateur &lt; 1 véh/h).<br/>
<b>MAE / RMSE</b> : en véh/h.
❄ = code source gelé (snapshot).
</p>
<footer>evaluate.py · {date}</footer></body></html>"""


def generate(results, output_dir, look_back, horizon, title="Comparaison"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(results)
    print("  figures …")
    b_bars = fig_bars(rows)
    b_box  = fig_box(results)
    b_ts   = fig_ts(results)
    b_sc   = fig_scatter(results)
    b_err  = fig_errors(results)

    configs = ""
    for r in results:
        cfg_str = json.dumps(r.get("config", {}), indent=2, ensure_ascii=False)
        configs += (f"<details><summary>{r['name']}</summary>"
                    f"<p style='font-size:.8rem;color:#888'>Checkpoint : {r.get('ckpt_path','—')}</p>"
                    f"<pre>{cfg_str}</pre></details>")

    schema_warn = ""
    for r in results:
        if r.get("schema_warn"):
            schema_warn += (f'<div class="schema-warn">⚠ <b>{r["name"]}</b> : '
                            f'{r["schema_warn"]}</div>')

    html = _HTML.format(
        title=title, css=_CSS,
        date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        nm=len(results), ns=results[0]["y_h"].shape[0],
        lb=look_back, hz=horizon,
        schema_warn=schema_warn, table=_table(rows),
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts",         nargs="*", default=[], metavar="FILE")
    p.add_argument("--snapshots",     nargs="*", default=[], metavar="DIR")
    p.add_argument("--baselines",     nargs="*", default=[], metavar="FILE")
    p.add_argument("--baseline_mode", default="dynamic_static",
                   choices=["dynamic_static","static_only","flow_only"])
    p.add_argument("--test_data",     default="data/test_data.parquet")
    p.add_argument("--scalers",       default=None)
    p.add_argument("--look_back",     type=int, default=None)
    p.add_argument("--horizon",       type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--output",        default="results/comparison")
    p.add_argument("--title",         default="Comparaison INR / LSTM")
    p.add_argument("--list",          metavar="DIR", default=None)
    p.add_argument("--device",        default=None)
    p.add_argument("--skip_compat",   action="store_true")
    args = p.parse_args()

    if args.list:
        from src.snapshot import list_snapshots, snapshot_info
        for s in list_snapshots(Path(args.list)):
            i     = snapshot_info(s)
            icon  = "❄ " if i["frozen_src"] else "  "
            schema_ok = "✓" if i.get("schema_ok", True) else "✗ SCHEMA INCOMPATIBLE"
            print(f"\n{icon}{i['name']}  [{schema_ok}]")
            print(f"  {i.get('saved_at','?')}  git:{i.get('git_hash','?')}")
            print(f"  look_back={i.get('look_back','?')} horizon={i.get('horizon','?')} "
                  f"control={i.get('control','?')} latent={i.get('latent_dim','?')}")
            if i.get("schema_diffs"):
                for d in i["schema_diffs"]: print(f"    {d}")
        return

    if not any([args.ckpts, args.snapshots, args.baselines]):
        p.error("Spécifie --ckpts, --snapshots ou --baselines")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")

    lb, hz = args.look_back, args.horizon
    if lb is None and args.ckpts:
        ckpt = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
        lb   = int(ckpt.get("cfg_data", {}).get("look_back_window", 192))
        hz   = int(ckpt.get("cfg_data", {}).get("horizon", 48))
    if lb is None and args.snapshots:
        c  = OmegaConf.load(Path(args.snapshots[0]) / "config.yaml")
        lb = int(OmegaConf.select(c, "data.look_back_window", default=192))
        hz = int(OmegaConf.select(c, "data.horizon", default=48))
    lb = lb or 192; hz = hz or 48
    print(f"look_back={lb}  horizon={hz}")

    scalers = None
    if args.scalers and Path(args.scalers).exists():
        scalers = joblib.load(args.scalers); print(f"Scalers : {args.scalers}")
    else:
        print("⚠ Pas de scalers → métriques en espace normalisé")

    latent_dim = 256
    if args.ckpts:
        ckpt       = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
        latent_dim = int(ckpt.get("cfg_inr", {}).get("latent_dim", 256))

    from src.dataloaders import NZDataset
    print(f"\nChargement : {args.test_data}")
    testset = NZDataset(ROOT / args.test_data, mode="test",
                        scalers=scalers, latent_dim=latent_dim)
    loader  = DataLoader(testset, batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), shuffle=False)
    print(f"Samples test : {len(testset)}")

    results = []

    for f in args.ckpts:
        print(f"\n→ INR legacy : {f}")
        try:
            r = evaluate_inr_ckpt(Path(f), loader, device, scalers)
            if r:
                results.append(r)
                m = metrics(r["y_h"], r["pred_h"])
                print(f"   MAE_h={m['MAE']:.2f}  RMSE_h={m['RMSE']:.2f}  sMAPE_h={m['sMAPE']:.1f}%")
        except Exception as e:
            print(f"   ✗ {e}"); import traceback; traceback.print_exc()

    for s in args.snapshots:
        print(f"\n→ INR snapshot : {s}")
        try:
            r = evaluate_inr_snapshot(Path(s), loader, device, scalers, args.skip_compat)
            if r:
                results.append(r)
                m = metrics(r["y_h"], r["pred_h"])
                print(f"   MAE_h={m['MAE']:.2f}  RMSE_h={m['RMSE']:.2f}  sMAPE_h={m['sMAPE']:.1f}%")
        except Exception as e:
            print(f"   ✗ {e}"); import traceback; traceback.print_exc()

    for bl in args.baselines:
        print(f"\n→ LSTM : {bl}")
        try:
            r = evaluate_baseline_lstm(Path(bl), loader, device, lb, scalers, args.baseline_mode)
            if r:
                results.append(r)
                m = metrics(r["y_h"], r["pred_h"])
                print(f"   MAE_h={m['MAE']:.2f}  RMSE_h={m['RMSE']:.2f}  sMAPE_h={m['sMAPE']:.1f}%")
        except Exception as e:
            print(f"   ✗ {e}"); import traceback; traceback.print_exc()

    if not results:
        print("\n✗ Aucun modèle évalué."); return

    print(f"\nGénération du rapport → {args.output}/")
    generate(results, Path(args.output), lb, hz, args.title)


if __name__ == "__main__":
    main()