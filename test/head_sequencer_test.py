import sys
import torch
import joblib
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.dataloaders import NZDataset
from src.head_sequencer import LSTM_HEAD

# ── Config ────────────────────────────────────────────────────
SAVE_DIR   = ROOT / 'save_models' / 'head_lstm'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'head_sequencer'
TWIN_IDX   = 120
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Chargement ────────────────────────────────────────────────
ckpt_file = sorted(SAVE_DIR.glob('*_best.pt'))[-1]
scalers   = joblib.load(SAVE_DIR / 'scalers.pkl')
print(f"Checkpoint : {ckpt_file.name}")

ckpt  = torch.load(ckpt_file, map_location=DEVICE)
model = LSTM_HEAD(context_dim=13, hidden_dim=128).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

# ── Dataset ───────────────────────────────────────────────────
testset = NZDataset(ROOT / 'data/test_data.parquet', mode='test', scalers=scalers)
loader  = DataLoader(testset, batch_size=2, shuffle=True)

# ── Inférence ─────────────────────────────────────────────────
t_pts, _, _, x_meteo, x_time, y, _ = next(iter(loader))
x_dyn = torch.cat([x_meteo, x_time, t_pts.float()], dim=-1).to(DEVICE)
y     = y.to(DEVICE)

with torch.no_grad():
    preds = model(x_dyn, y, twin_idx=TWIN_IDX)   # (B, T-1, 1)

# ── Dénormalisation ───────────────────────────────────────────
std  = scalers['target'].std.squeeze()
mean = scalers['target'].mean.squeeze()
eps  = scalers['target'].eps

y_np    = (y[:, 1:, 0].cpu().numpy()    * (std + eps) + mean)
pred_np = (preds[:, :, 0].cpu().numpy() * (std + eps) + mean)

# ── Print ──────────────────────────────────────────────────────
for i in range(2):
    print(f"\n── Exemple {i+1} ──────────────────────────────────")
    print(f"  [serie_p] target : {y_np[i,    :8].round(1)}")
    print(f"  [serie_p] prédit : {pred_np[i,  :8].round(1)}")
    print(f"  [serie_h] target : {y_np[i,    TWIN_IDX:TWIN_IDX+8].round(1)}")
    print(f"  [serie_h] prédit : {pred_np[i, TWIN_IDX:TWIN_IDX+8].round(1)}")

# ── HTML ───────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for i in range(2):
    T       = len(y_np[i])
    labels  = list(range(T))

    obs_p   = y_np[i,    :TWIN_IDX].tolist()
    obs_h   = y_np[i,    TWIN_IDX:].tolist()
    pred_p  = pred_np[i, :TWIN_IDX].tolist()
    pred_h  = pred_np[i, TWIN_IDX:].tolist()

    # Padding pour aligner sur T points
    pad_p  = [None] * (T - TWIN_IDX)
    pad_h  = [None] * TWIN_IDX

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Exemple {i+1}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body  {{ font-family: sans-serif; background: #f5f5f5; padding: 2rem; }}
    .card {{ background: white; border-radius: 12px; padding: 1.5rem;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 960px; margin-bottom: 2rem; }}
    h2    {{ margin: 0 0 .4rem; color: #222; }}
    .meta {{ font-size: 12px; color: #999; margin-bottom: 1.2rem; }}
    .legend-box {{ display: flex; gap: 1.5rem; font-size: 13px; margin-bottom: 1rem; }}
    .dot  {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  </style>
</head>
<body>
<div class="card">
  <h2>Exemple {i+1} — LSTM HEAD</h2>
  <p class="meta">
    Checkpoint : {ckpt_file.name} &nbsp;|&nbsp;
    Epoch : {ckpt['epoch']} &nbsp;|&nbsp;
    Val loss : {ckpt['val_loss']:.4f} &nbsp;|&nbsp;
    Val loss P : {ckpt['val_loss_p']:.4f} &nbsp;|&nbsp;
    Val loss H : {ckpt['val_loss_h']:.4f} &nbsp;|&nbsp;
    Twin idx : {TWIN_IDX}
  </p>
  <div class="legend-box">
    <span><span class="dot" style="background:#4e79a7"></span>Cible serie_p (observation)</span>
    <span><span class="dot" style="background:#f28e2b"></span>Prédit serie_p</span>
    <span><span class="dot" style="background:#59a14f"></span>Cible serie_h (horizon)</span>
    <span><span class="dot" style="background:#e15759"></span>Prédit serie_h</span>
  </div>
  <canvas id="chart{i}" height="90"></canvas>
</div>
<script>
new Chart(document.getElementById('chart{i}'), {{
  type: 'line',
  data: {{
    labels: {labels},
    datasets: [
      {{
        label: 'Cible serie_p',
        data: {obs_p + pad_p},
        borderColor: '#4e79a7', borderWidth: 2,
        pointRadius: 0, tension: 0.3,
      }},
      {{
        label: 'Prédit serie_p',
        data: {pred_p + pad_p},
        borderColor: '#f28e2b', borderWidth: 2,
        borderDash: [4, 4], pointRadius: 0, tension: 0.3,
      }},
      {{
        label: 'Cible serie_h',
        data: {pad_h + obs_h},
        borderColor: '#59a14f', borderWidth: 2,
        pointRadius: 0, tension: 0.3,
      }},
      {{
        label: 'Prédit serie_h',
        data: {pad_h + pred_h},
        borderColor: '#e15759', borderWidth: 2,
        borderDash: [4, 4], pointRadius: 0, tension: 0.3,
      }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Heure' }},
        ticks: {{ maxTicksLimit: 20 }},
      }},
      y: {{ title: {{ display: true, text: 'Flow (véh/h)' }} }},
    }},
    annotation: {{
      annotations: {{
        twin: {{
          type: 'line',
          x: {TWIN_IDX},
          borderColor: '#aaa',
          borderWidth: 1,
          borderDash: [6, 3],
          label: {{ content: 'twin', display: true, position: 'start' }}
        }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out = OUTPUT_DIR / f"exemple_{i+1}.html"
    out.write_text(html, encoding='utf-8')
    print(f"Sauvegardé : {out}")