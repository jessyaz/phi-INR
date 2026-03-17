from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
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

def load_checkpoint(cfg) -> tuple[dict, Path]:
    """Charge le checkpoint spécifié ou le dernier sauvegardé."""
    save_dir = ROOT / cfg.paths.save_models

    manual = cfg.get('inference', {}).get('ckpt', None)
    if manual:
        ckpt_path = Path(manual)
    else:
        candidates = sorted(save_dir.glob('*_best.pt'))
        if not candidates:
            raise FileNotFoundError(f"Aucun checkpoint trouvé dans {save_dir}")
        ckpt_path = candidates[-1]

    print(f"Checkpoint : {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  epoch      : {ckpt['epoch']}")
    print(f"  val_loss   : {ckpt['val_loss']:.6f}")
    print(f"  val_loss_p : {ckpt['val_loss_p']:.6f}")
    print(f"  val_loss_h : {ckpt['val_loss_h']:.6f}")
    return ckpt, ckpt_path


def build_model_from_checkpoint(ckpt: dict, cfg: DictConfig, device) -> ModulatedFourierFeatures:
    inr_cfg  = OmegaConf.create(ckpt['cfg_inr'])
    data_cfg = OmegaConf.create(ckpt['cfg_data'])

    assert data_cfg.look_back_window == cfg.data.look_back_window, (
        f"look_back_window mismatch : ckpt={data_cfg.look_back_window} "
        f"config={cfg.data.look_back_window}"
    )

    inr = ModulatedFourierFeatures(
        input_dim        = cfg.data.input_dim,
        output_dim       = cfg.data.output_dim,
        look_back_window = cfg.data.look_back_window,
        num_frequencies  = inr_cfg.num_frequencies,
        latent_dim       = inr_cfg.latent_dim,
        lstm_hidden_dim  = inr_cfg.lstm_hidden_dim,
        spatial_dim      = inr_cfg.static.spatial_dim,
        dir_dim          = inr_cfg.static.dir_dim,
        num_directions   = inr_cfg.static.num_directions,
        sigma            = inr_cfg.static.sigma,
        width            = inr_cfg.hidden_dim,
        depth            = inr_cfg.depth,
        min_frequencies  = inr_cfg.min_frequencies,
        base_frequency   = inr_cfg.base_frequency,
        include_input    = inr_cfg.include_input,
        is_training      = False,
        use_context      = inr_cfg.use_context,
        freeze_lstm      = False,
    )

    inr.load_state_dict(ckpt['inr_state_dict'])
    inr = inr.to(device).eval()
    print(f"Modèle chargé — mode : {'context LSTM' if inr_cfg.use_context else 'INR vanilla'}")
    return inr


def build_html_report(plot_samples, all_mae, cfg, ckpt, ckpt_path, look_back, horizon) -> str:
    use_context = cfg.inr.use_context
    n      = len(plot_samples)
    label  = "avec contexte LSTM" if use_context else "INR vanilla"
    colors = {"past": "royalblue", "target": "seagreen", "forecast": "crimson"}

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False,
        subplot_titles=[f"Sample {i+1}  —  MAE={mae(s['target'], s['forecast']):.4f}"
                        for i, s in enumerate(plot_samples)],
        vertical_spacing=0.06,
    )

    for i, s in enumerate(plot_samples):
        row = i + 1
        x_p = np.arange(len(s['past']))
        x_h = np.arange(len(s['past']), len(s['past']) + len(s['target']))

        fig.add_trace(go.Scatter(x=x_p, y=s['past'], mode='lines',
                                 name='Passé', legendgroup='past',
                                 showlegend=(i == 0),
                                 line=dict(color=colors['past'], width=1.5)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x_p, y=s['pred_p'], mode='lines',
                                 name='Reconstruit P', legendgroup='pred_p',
                                 showlegend=(i == 0),
                                 line=dict(color='orange', width=1.2, dash='dot')),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x_h, y=s['target'], mode='lines',
                                 name='Cible', legendgroup='target',
                                 showlegend=(i == 0),
                                 line=dict(color=colors['target'], width=1.5)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=x_h, y=s['forecast'], mode='lines',
                                 name='Prédiction', legendgroup='forecast',
                                 showlegend=(i == 0),
                                 line=dict(color=colors['forecast'], width=1.5, dash='dash')),
                      row=row, col=1)
        fig.add_vline(x=len(s['past']) - 0.5,
                      line_dash='dot', line_color='grey', opacity=0.5,
                      row=row, col=1)

    fig.update_layout(
        height=350 * n,
        title=dict(text=(
            f"<b>Inférence — {cfg.data.dataset_name} ({label})</b><br>"
            f"<span style='font-size:13px'>"
            f"Mean MAE : {np.mean(all_mae):.6f}  |  "
            f"look-back : {look_back}  |  horizon : {horizon}  |  "
            f"inner steps : {cfg.inner.inner_steps}  |  "
            f"ckpt : {ckpt_path.name}</span>"
        ), x=0.5),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        template='plotly_white',
    )

    plt_div   = plot(fig, output_type='div', include_plotlyjs=True)
    badge_bg  = '#d4edda' if use_context else '#fff3cd'
    badge_col = '#155724' if use_context else '#856404'

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Inférence — {cfg.data.dataset_name}</title>
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
</html>"""


# ── Main ─────────────────────────────────────────────────────

@hydra.main(config_path="conf/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    look_back   = cfg.data.look_back_window
    horizon     = cfg.data.horizon
    use_context = cfg.inr.use_context
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device      : {device}")
    print(f"use_context : {use_context}")

    ckpt, ckpt_path = load_checkpoint(cfg)

    scalers = joblib.load(ROOT / cfg.paths.scalers_file)
    print(f"Scalers chargés : {ROOT / cfg.paths.scalers_file}")

    # ── Dataset ──────────────────────────────────────────────
    testset = NZDataset(
        parquet_file = ROOT / cfg.data.test_parquet,
        mode         = 'test',
        scalers      = scalers,
        latent_dim   = cfg.inr.latent_dim,
    )
    test_loader = DataLoader(testset, batch_size=cfg.optim.batch_size,
                             shuffle=False, num_workers=0)

    # ── Modèle ───────────────────────────────────────────────
    inr   = build_model_from_checkpoint(ckpt, cfg, device)
    alpha = torch.tensor([cfg.optim.lr_code], device=device)

    # ── Inférence ────────────────────────────────────────────
    all_mae      = []
    plot_samples = []
    n_plots_max  = cfg.get('inference', {}).get('n_plots', 8)

    for batch in test_loader:
        t, dir_idx, x_static, x_meteo, x_time, y, code = batch

        x_context = torch.cat([x_meteo, x_time], dim=-1).to(device)
        y_target  = y.to(device)
        t         = t.to(device)

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
            is_train    = False,
            code        = torch.zeros(t.shape[0], cfg.inr.latent_dim, device=device),
            x_statics   = x_statics,
            dir_idx     = dir_idx,
        )

        out_p = outputs['out_p']
        out_h = outputs['out_h']

        # ── Dénormalisation ───────────────────────────────────
        scaler = scalers['target']
        std    = scaler.std.squeeze()
        mean   = scaler.mean.squeeze()
        eps    = scaler.eps

        def denorm(arr):
            return arr * (std + eps) + mean

        y_p_np  = denorm(y_past   [:, :, 0].cpu().numpy())
        y_np    = denorm(y_horizon[:, :, 0].cpu().numpy())
        pred_p  = denorm(out_p    [:, :, 0].cpu().numpy())
        fc_np   = denorm(out_h    [:, :, 0].cpu().numpy()) \
            if out_h is not None else np.zeros_like(y_np)

        # ── Debug premier batch ───────────────────────────────
        if len(all_mae) == 0:
            print("\n── Debug premier batch ──────────────────────────")
            for i in range(min(3, y_p_np.shape[0])):
                print(f"\n  Sample {i+1}")
                print(f"  passé   (8 derniers) : {y_p_np[i, -8:].round(1)}")
                print(f"  pred_p  (8 derniers) : {pred_p[i,  -8:].round(1)}")
                print(f"  cible   (8 premiers) : {y_np[i,    :8].round(1)}")
                print(f"  prédit  (8 premiers) : {fc_np[i,   :8].round(1)}")
                print(f"  cible   min/max      : {y_np[i].min():.2f} / {y_np[i].max():.2f}")
                print(f"  prédit  min/max      : {fc_np[i].min():.2f} / {fc_np[i].max():.2f}")
            print("─────────────────────────────────────────────────\n")

        all_mae.append(mae(y_np.reshape(-1), fc_np.reshape(-1)))

        if len(plot_samples) < n_plots_max:
            n_take  = min(n_plots_max - len(plot_samples), t.shape[0])
            indices = torch.randperm(t.shape[0])[:n_take].tolist()
            for i in indices:
                plot_samples.append({
                    'past':     y_p_np[i],
                    'pred_p':   pred_p[i],    # ← ajout
                    'target':   y_np[i],
                    'forecast': fc_np[i],
                })
    print(f"\nMean MAE : {np.mean(all_mae):.6f}")

    ctx_tag  = 'ctx' if use_context else 'noctx'
    out_html = ROOT / f"inference_{cfg.data.dataset_name}_h{horizon}_{ctx_tag}.html"
    html     = build_html_report(plot_samples, all_mae, cfg, ckpt, ckpt_path, look_back, horizon)
    out_html.write_text(html, encoding='utf-8')
    print(f"Rapport → {out_html}")


if __name__ == "__main__":
    main()