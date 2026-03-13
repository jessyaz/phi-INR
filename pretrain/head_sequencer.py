import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.dataloaders import NZDataset

class LSTM_HEAD(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # context(13) + flow(1) + h_since_tw(1) = 15
        self.input_dim = context_dim + 1 + 1

        self.W = nn.Parameter(torch.randn(self.input_dim + hidden_dim, hidden_dim * 4) * 0.02)
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))
        self.out_flow = nn.Linear(hidden_dim, 1)

    def cell_step(self, x_t, h_t, c_t):
        combined = torch.cat((x_t, h_t), dim=1)
        gates = combined @ self.W + self.b
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def forward(self, x_dyn, y_flow, twin_idx):
        bs, seq_len, _ = x_dyn.size()
        h_t = torch.zeros(bs, self.hidden_dim, device=x_dyn.device)
        c_t = torch.zeros(bs, self.hidden_dim, device=x_dyn.device)
        preds = []

        # 1. Observation
        for t in range(twin_idx):
            h_since_tw = torch.zeros(bs, 1, device=x_dyn.device)
            step_input = torch.cat([x_dyn[:, t, :], y_flow[:, t, :], h_since_tw], dim=-1)
            h_t, c_t = self.cell_step(step_input, h_t, c_t)
            preds.append(self.out_flow(h_t).unsqueeze(1))

        # 2. Auto-régressif avec H_SINCE_TW
        last_out = preds[-1][:, 0, :]
        for t in range(twin_idx, seq_len - 1):
            # Temps écoulé depuis le début de la prédiction (normalisé par 24h)
            h_val = (t - twin_idx + 1) / 24.0
            h_since_tw = torch.full((bs, 1), h_val, device=x_dyn.device)

            step_input = torch.cat([x_dyn[:, t, :], last_out, h_since_tw], dim=-1)
            h_t, c_t = self.cell_step(step_input, h_t, c_t)
            last_out = self.out_flow(h_t)
            preds.append(last_out.unsqueeze(1))

        return torch.cat(preds, dim=1)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    conf = cfg.head_lstm

    # Debug: Vérifie si le dataset est vide
    trainset = NZDataset(
        parquet_file = ROOT / conf.data.train_parquet,
        window       = f"{conf.data.num_days}D",
        latent_dim   = conf.inr.latent_dim,
    )

    if len(trainset) == 0:
        print("ERREUR: Le dataset est vide. Vérifie les NaNs dans 'FLOW' ou la taille de 'window'.")
        return

    train_loader = DataLoader(trainset, batch_size=conf.data.batch_size, shuffle=True)

    # Input_dim passe à 13 (meteo+time+t)
    model = LSTM_HEAD(13, conf.model.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=conf.model.lr)
    criterion = nn.MSELoss()

    mlflow.set_experiment(conf.mlflow.experiment_name)

    with mlflow.start_run():
        for epoch in range(10):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                t_pts, _, _, x_meteo, x_time, y, _ = batch
                x_dyn = torch.cat([x_meteo, x_time, t_pts.float()], dim=-1)

                optimizer.zero_grad()
                preds = model(x_dyn, y, twin_idx=conf.model.twin_idx)

                # y est (B, T, 1), on prédit de t=1 à T
                loss = criterion(preds, y[:, 1:, :])

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()