from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import hydra
from sklearn.metrics import mean_absolute_error as mae
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.dataloaders import NZDataset
from src.metalearning import outer_step
from src.network import ModulatedFourierFeatures


# ── Helpers ──────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Checkpoint chargé : epoch {ckpt['epoch']}, train_loss {ckpt['train_loss']:.6f}")
    return ckpt


def build_model_from_checkpoint(ckpt: dict, cfg: DictConfig, device):
    inr_cfg  = OmegaConf.create(ckpt["cfg_inr"])
    data_cfg = OmegaConf.create(ckpt["cfg_data"])

    assert data_cfg.look_back_window == cfg.data.look_back_window, (
        f"look_back_window mismatch : checkpoint={data_cfg.look_back_window}, config={cfg.data.look_back_window}"
    )

    inr = ModulatedFourierFeatures(
        input_dim        = cfg.data.input_dim,
        output_dim       = cfg.data.output_dim,
        x_dyn_c_dim      = cfg.data.x_dyn_c_dim,
        x_stat_dim       = cfg.data.x_stat_dim,
        look_back_window = cfg.data.look_back_window,
        num_frequencies  = inr_cfg.num_frequencies,
        latent_dim       = inr_cfg.latent_dim,
        static_emb_dim   = inr_cfg.static_emb_dim,
        width            = inr_cfg.hidden_dim,
        depth            = inr_cfg.depth,
        min_frequencies  = inr_cfg.min_frequencies,
        base_frequency   = inr_cfg.base_frequency,
        include_input    = inr_cfg.include_input,
        is_training      = False,
        use_context      = cfg.inr.use_context,
    )

    sd = ckpt.get("inr_state_dict")
    if sd is None:
        raise KeyError(f"'inr_state_dict' introuvable. Clés disponibles : {list(ckpt.keys())}")

    inr.load_state_dict(sd)
    inr = inr.to(device).eval()
    mode = "avec contexte (VAE)" if cfg.inr.use_context else "sans contexte (INR seul)"
    print(f"Modèle chargé en mode : {mode}")
    return inr


def build_html_report(plot_samples, all_mae, cfg, ckpt, look_back, horizon) -> str:
    use_context = cfg.inr.use_context
    n      = len(plot_samples)
    label  = "avec contexte" if use_context else "sans contexte"
    colors = {"past": "royalblue", "target": "seagreen", "forecast": "crimson"}

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False,
        subplot_titles=[f"Sample {i + 1}" for i in range(n)],
        vertical_spacing=0.06,
    )

    for i, s in enumerate(plot_samples):
        row   = i + 1
        x_p   = np.arange(len(s["past"]))
        x_h   = np.arange(len(s["past"]), len(s["past"]) + len(s["target"]))
        s_mae = mae(s["target"], s["forecast"])

        fig.add_trace(go.Scatter(x=x_p, y=s["past"], mode="lines",
                                 name="Passé", legendgroup="past",
                                 showlegend=(i == 0), line=dict(color=colors["past"], width=1.5)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x_h, y=s["target"], mode="lines",
                                 name="Cible", legendgroup="target",
                                 showlegend=(i == 0), line=dict(color=colors["target"], width=1.5)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x_h, y=s["forecast"], mode="lines",
                                 name="Prédiction", legendgroup="forecast",
                                 showlegend=(i == 0), line=dict(color=colors["forecast"], width=1.5, dash="dash")),
                      row=row, col=1)
        fig.add_vline(x=len(s["past"]) - 0.5, line_dash="dot", line_color="grey", opacity=0.5, row=row, col=1)
        fig.add_annotation(xref="x domain", yref="y domain", x=0.99, y=0.97,
                           text=f"MAE = {s_mae:.4f}", showarrow=False,
                           font=dict(size=11), bgcolor="rgba(255,255,255,0.7)",
                           align="right", row=row, col=1)

    fig.update_layout(
        height=350 * n,
        title=dict(text=(
            f"<b>Inférence – {cfg.data.dataset_name} ({label})</b><br>"
            f"<span style='font-size:14px'>"
            f"Mean MAE : {np.mean(all_mae):.6f} | "
            f"look-back : {look_back} | horizon : {horizon} | "
            f"inner steps : {cfg.inner.inner_steps}</span>"
        ), x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        template="plotly_white",
    )

    plt_div   = plot(fig, output_type="div", include_plotlyjs=True)
    badge_bg  = "#d4edda" if use_context else "#fff3cd"
    badge_col = "#155724" if use_context else "#856404"

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Inférence – {cfg.data.dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
        .summary {{
            background: white; border-radius: 8px; padding: 16px 24px;
            margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
            display: flex; gap: 40px; flex-wrap: wrap; align-items: center;
        }}
        .metric {{ display: flex; flex-direction: column; }}
        .metric span:first-child {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric span:last-child  {{ font-size: 22px; font-weight: bold; color: #333; }}
        .badge {{
            background: {badge_bg}; color: {badge_col};
            padding: 6px 14px; border-radius: 12px;
            font-weight: bold; font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="summary">
        <div class="metric"><span>Dataset</span><span>{cfg.data.dataset_name}</span></div>
        <div class="metric"><span>Mean MAE</span><span>{np.mean(all_mae):.6f}</span></div>
        <div class="metric"><span>Look-back</span><span>{look_back}</span></div>
        <div class="metric"><span>Horizon</span><span>{horizon}</span></div>
        <div class="metric"><span>Inner steps</span><span>{cfg.inner.inner_steps}</span></div>
        <div class="metric"><span>Epoch</span><span>{ckpt['epoch']}</span></div>
        <div class="badge">{label}</div>
    </div>
    {plt_div}
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────

@hydra.main(config_path="conf/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    look_back   = cfg.data.look_back_window
    horizon     = cfg.data.horizon
    use_context = cfg.inr.use_context

    model_path = (
        f"{cfg.paths.save_models}/models_forecasting"
        f"_{cfg.data.dataset_name}_{horizon}_{use_context}"
        f"_{cfg.meta.experiment_pack}_{cfg.optim.epochs}_{cfg.misc.version}.pt"
    )
    ckpt = load_checkpoint(model_path)

    scaler_path = ROOT / cfg.paths.scalers_file
    scalers     = joblib.load(scaler_path) if scaler_path.exists() else None
    if scalers is None:
        print("WARNING : scalers introuvables.")

    # ── Dataset ──────────────────────────────────────────────
    testset = NZDataset(
        parquet_file = ROOT / cfg.data.test_parquet,
        num_days     = cfg.data.num_days,
        mode         = "test" if scalers is not None else "train",
        scalers      = scalers,
        latent_dim   = cfg.inr.latent_dim,
    )
    test_loader = DataLoader(
        testset,
        batch_size  = cfg.optim.batch_size,
        shuffle     = False,
        num_workers = cfg.data.num_workers,
        pin_memory  = cfg.data.pin_memory,
    )

    # ── Modèle ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    inr   = build_model_from_checkpoint(ckpt, cfg, device)
    alpha = torch.tensor([cfg.optim.lr_code], device=device)

    # ── Inférence ────────────────────────────────────────────
    all_mae      = []
    plot_samples = []
    n_plots_max  = cfg.misc.get("n_plots", 5)

    for x_time, x_statics, x_dynamics, y_target, modulations in test_loader:
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
            is_train    = False,
            modulations = torch.zeros_like(modulations),
        )

        modulations_opt = outputs["modulations"].detach()

        with torch.no_grad():
            forecast, _ = inr.modulated_forward(coords_h, modulations_opt, features, x_statics)

        # ── Inverse transform ────────────────────────────────
        y_np   = y_horizon.cpu().numpy()
        fc_np  = forecast.cpu().numpy()
        if scalers is not None:
            y_np  = scalers["Y_target"].inverse_transform(y_np.reshape(-1, 1)).reshape(y_np.shape)
            fc_np = scalers["Y_target"].inverse_transform(fc_np.reshape(-1, 1)).reshape(fc_np.shape)

        all_mae.append(mae(y_np.reshape(-1), fc_np.reshape(-1)))

        if len(plot_samples) < n_plots_max:
            n_take     = min(n_plots_max - len(plot_samples), x_time.shape[0])
            indices    = torch.randperm(x_time.shape[0])[:n_take].tolist()
            y_past_np  = y_past.cpu().numpy()
            if scalers is not None:
                y_past_np = scalers["Y_target"].inverse_transform(
                    y_past_np.reshape(-1, 1)
                ).reshape(y_past_np.shape)

            for i in indices:
                plot_samples.append({
                    "past":     y_past_np[i, :, 0],
                    "target":   y_np[i, :, 0],
                    "forecast": fc_np[i, :, 0],
                })

    print(f"\nMean MAE : {np.mean(all_mae):.6f}")

    ctx_tag  = "ctx" if use_context else "noctx"
    out_html = ROOT / f"inference_{cfg.data.dataset_name}_h{horizon}_{ctx_tag}.html"
    html     = build_html_report(plot_samples, all_mae, cfg, ckpt, look_back, horizon)
    out_html.write_text(html, encoding="utf-8")
    print(f"Rapport sauvegardé → {out_html}")


if __name__ == "__main__":
    main()