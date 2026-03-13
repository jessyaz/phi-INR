import torch
import torch.nn as nn
import torch.nn.functional as F
from src.film_conditionning import film_translate

# ── Modules auxiliaires ──────────────────────────────────────
class FourierSpatialEncoder(nn.Module):
    def __init__(self, num_freqs=16, d_out=32, sigma=1.0):
        super().__init__()
        # sigma contrôle l'échelle : petit = sensible aux grandes distances,
        #                             grand = sensible aux petites distances
        B = torch.randn(2, num_freqs) * sigma
        self.register_buffer('B', B)

        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

    def forward(self, latlon):
        proj  = 2 * torch.pi * latlon @ self.B   # (B, num_freqs)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, 2*num_freqs)
        return self.mlp(feats)


class StaticEncoder(nn.Module):
    def __init__(self, x_stat_dim, static_emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_stat_dim, static_emb_dim),
            nn.SiLU(),
        )

    def forward(self, x_statics):
        return self.net(x_statics.float())


class LatentToModulation(nn.Module):
    def __init__(self, latent_dim, static_emb_dim, num_modulations):
        super().__init__()
        self.net = nn.Linear(latent_dim * 2 + static_emb_dim, num_modulations)

    def forward(self, modulation, z_lat, static_emb):
        return self.net(torch.cat([modulation, z_lat, static_emb], dim=1))


# ── Encodage positionnel (NeRF) ──────────────────────────────

class NeRFEncoding(nn.Module):
    def __init__(self, num_frequencies, min_freq, include_input=True, input_dim=1, base_freq=1.25):
        super().__init__()
        self.include_input = include_input

        bands = base_freq * torch.arange(min_freq, num_frequencies, 1, dtype=torch.float32)
        self.bands   = nn.Parameter(bands, requires_grad=False)
        self.out_dim = bands.shape[0] * input_dim * 2
        if include_input:
            self.out_dim += input_dim

    def forward(self, coords):
        # coords : (B, T, input_dim)
        winded  = (coords[..., None, :] * self.bands[None, None, :, None])
        winded  = winded.reshape(coords.shape[0], coords.shape[1], -1)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded


# ── Modèle principal ─────────────────────────────────────────

class ModulatedFourierFeatures(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            x_dyn_c_dim,
            x_stat_dim,
            look_back_window,
            num_frequencies=8,
            latent_dim=128,
            static_emb_dim=16,
            width=256,
            depth=3,
            min_frequencies=0,
            base_frequency=1.25,
            include_input=True,
            is_training=True,
            use_context=True,
    ):
        super().__init__()

        self.is_training = is_training   # NOTE: ne pas utiliser self.training (attribut PyTorch réservé)
        self.use_context = use_context
        self.latent_dim  = latent_dim

        self.embedding = NeRFEncoding(
            num_frequencies = num_frequencies,
            min_freq        = min_frequencies,
            include_input   = include_input,
            input_dim       = input_dim,
            base_freq       = base_frequency,
        )

        in_channels  = [self.embedding.out_dim] + [width] * (depth - 1)
        out_channels = [width] * (depth - 1) + [output_dim]
        self.layers  = nn.ModuleList([nn.Linear(i, o) for i, o in zip(in_channels, out_channels)])
        self.depth   = depth

        num_modulations = width * (depth - 1)

        self.vae = VAE(
            input_dim  = x_dyn_c_dim,
            latent_dim = latent_dim,
            seq_len    = look_back_window,
        )
        self.static_encoder      = StaticEncoder(x_stat_dim, static_emb_dim)
        self.latent_to_modulation = LatentToModulation(latent_dim, static_emb_dim, num_modulations)

    def set_context_mode(self, use_context: bool) -> None:
        self.use_context = use_context
        mode = "with context (VAE)" if use_context else "without context (INR vanilla)"
        print(f"[ModulatedFourierFeatures] Mode : {mode}")

    def modulated_forward(self, coords, modulation, x_past, x_statics, beta=0):
        batch_size = coords.shape[0]

        if self.use_context:
            mu, logvar   = self.vae.encode(x_past)
            z_lat        = mu if not self.is_training else self.vae.reparameterize(mu, logvar)
            recon_x_past = self.vae.decode(z_lat)
            loss_vae     = self.vae.loss_function(recon_x_past, x_past, mu, logvar, beta=beta)
        else:
            z_lat    = torch.zeros(batch_size, self.latent_dim, device=coords.device, dtype=coords.dtype)
            loss_vae = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        static_emb  = self.static_encoder(x_statics)
        position    = self.embedding(coords)
        modulations = self.latent_to_modulation(modulation, z_lat, static_emb)
        pre_out     = film_translate(position, modulations, self.layers[:-1], torch.relu)
        out         = self.layers[-1](pre_out)

        return out, loss_vae


# ── VAE ──────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len):
        super().__init__()
        self.seq_len     = seq_len
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        flatten_dim      = input_dim * seq_len

        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256),         nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, flatten_dim),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder_net(z).view(-1, self.seq_len, self.input_dim)

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-4):
        mse = F.mse_loss(recon_x, x, reduction='mean') / x.size(0)
        kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return mse + beta * kl