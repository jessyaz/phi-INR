import torch
import torch.nn as nn
from src.film_conditionning import film_translate, film_translate_spe
from src.head_sequencer import LSTM_HEAD
from src.dataloaders import LSTM_IN_DIM, STAT_DIM


# ── Encodeur spatial Fourier ──────────────────────────────────

class FourierSpatialEncoder(nn.Module):
    def __init__(self, num_freqs=16, d_out=32, sigma=0.1):
        super().__init__()
        B = torch.randn(2, num_freqs) * sigma
        self.register_buffer('B', B)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

    def forward(self, latlon):
        proj  = 2 * torch.pi * latlon @ self.B
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(feats)


# ── Encodeur statique complet ─────────────────────────────────

class StaticEncoder(nn.Module):
    def __init__(self, input_dim=4, spatial_dim=32, dir_dim=8, num_directions=6, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),   # ← dynamique
            nn.GELU(),
            nn.Linear(32, spatial_dim),
        )
        self.dir_embedding = nn.Embedding(num_embeddings=num_directions, embedding_dim=dir_dim)
        self.out_dim       = spatial_dim + dir_dim

    def forward(self, x_statics, dir_idx):

        spatial = self.mlp(x_statics.float())

        if dir_idx is None:
            return spatial

        dir_vec = self.dir_embedding(dir_idx).unsqueeze(1).repeat(1,spatial.size(1),1)

        return torch.cat([spatial, dir_vec], dim=-1)

# ── Hypernetwork avec contexte ────────────────────────────────

class LatentToModulation(nn.Module):
    def __init__(self, latent_dim, lstm_hidden_dim, static_emb_dim, num_modulations, control=None):
        super().__init__()
        self.control = control

        if control == "static_only":
            input_dim = latent_dim + static_emb_dim
            print("HN Mode : ", control)
        elif control == "dynamic_only":
            input_dim = latent_dim + lstm_hidden_dim
            print("HN Mode : ", control)
        else:
            print("HN Mode : ", control)
            input_dim = latent_dim + static_emb_dim + lstm_hidden_dim


        self.net = nn.Sequential(nn.Linear(input_dim, num_modulations))


    def forward(self, code, h_t, static_emb):
        if self.control == "static_only":
            return self.net(torch.cat([code, static_emb], dim=-1))
        elif self.control == "dynamic_only":
            return self.net(torch.cat([code, h_t], dim=-1))
        else:

            return self.net(torch.cat([code, h_t, static_emb], dim=-1))




# ── Hypernetwork vanilla ──────────────────────────────────────

class LatentToModulationVanilla(nn.Module):
    """code seul → modulations (INR vanilla)"""

    def __init__(self, latent_dim, num_modulations, control=None):
        super().__init__()

        self.net = nn.Linear(latent_dim, num_modulations)

        if control is None:
            print("HN Mode :  vanilla")

    def forward(self, code):
        return self.net(code)


# ── Encodage positionnel NeRF ─────────────────────────────────

class NeRFEncoding(nn.Module):
    def __init__(self, num_frequencies, min_freq, input_dim=1, base_freq=1.25):
        super().__init__()

        bands              = base_freq * torch.arange(min_freq, num_frequencies, dtype=torch.float32)
        #self.bands         = nn.Parameter(bands, requires_grad=False)
        self.register_buffer('bands', bands)

        self.out_dim       = bands.shape[0] * input_dim * 2

    def forward(self, coords):
        winded  = coords[..., None, :] * self.bands[None, None, :, None]
        winded  = winded.reshape(*coords.shape[:2], -1)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)

        return encoded


# ── Modèle principal ──────────────────────────────────────────

class ModulatedFourierFeatures(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            look_back_window,
            num_frequencies  = 8,
            latent_dim       = 128,
            lstm_hidden_dim  = 128,
            spatial_dim      = 32,
            dir_dim          = 8,
            num_directions   = 6,
            sigma            = 0.1,
            width            = 256,
            depth            = 3,
            min_frequencies  = 0,
            base_frequency   = 1.25,
            is_training      = True,
            use_context      = True,
            freeze_lstm      = False,
            control          = None,
    ):
        super().__init__()
        self.is_training     = is_training
        self.use_context     = use_context
        self.latent_dim      = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.look_back       = look_back_window


        self._debug = False

        self.embedding = NeRFEncoding(
            num_frequencies = num_frequencies,
            min_freq        = min_frequencies,
            input_dim       = input_dim,
            base_freq       = base_frequency,
        )

        in_channels  = [self.embedding.out_dim] + [width] * (depth - 1)
        out_channels = [width] * (depth - 1) + [output_dim]

        print("CTE", in_channels, out_channels)

        self.layers  = nn.ModuleList([nn.Linear(i, o) for i, o in zip(in_channels, out_channels)])

        num_modulations = width * (depth - 1)

        self.num_modulations = num_modulations

        self.control = control

        if self.control != "static_only":
            # ── LSTM encodeur de contexte ──
            self.lstm = LSTM_HEAD(
                context_dim = LSTM_IN_DIM,
                hidden_dim  = lstm_hidden_dim,


            )
            if freeze_lstm:
                for p in self.lstm.parameters():
                    p.requires_grad_(False)
                print("[MFF] Poids LSTM gelés")

        # ── Encodeur statique ──
        self.static_encoder = StaticEncoder(
            input_dim      = 4,
            spatial_dim    = spatial_dim,
            dir_dim        = dir_dim,
            num_directions = num_directions,
            sigma          = sigma,
        )

        static_emb_dim = self.static_encoder.out_dim   # spatial_dim + dir_dim

        # ── Hypernetworks ──
        self.latent_to_mod = LatentToModulation(
            latent_dim      = latent_dim,
            lstm_hidden_dim = lstm_hidden_dim,
            static_emb_dim  = static_emb_dim,
            num_modulations = num_modulations,
            control = control,
        )

        self.latent_to_mod_vanilla = LatentToModulationVanilla(
            latent_dim      = latent_dim,
            num_modulations = num_modulations,
            control         = control,
        )


    def load_lstm_weights(self, ckpt_path: str, device='cpu'):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.lstm.W.data = ckpt['model']['W']
        self.lstm.b.data = ckpt['model']['b']
        print(f"Poids LSTM chargés — W:{self.lstm.W.shape} b:{self.lstm.b.shape}")
        print(f"Depuis : {ckpt_path}")

    def set_context_mode(self, use_context: bool):
        self.use_context = use_context
        print(f"[MFF] Mode : {'context LSTM' if use_context else 'INR vanilla'}")

    def _inr_batch(self, coords, code, hs, x_statics, dir_idx):
        B, T, _ = coords.shape
        #static_emb = self.static_encoder(x_statics, dir_idx)          # (B, S)
        static_emb = self.static_encoder(x_statics, dir_idx) if x_statics is not None else None
        position   = self.embedding(coords)                            # (B, T, pos_dim)
        out = torch.zeros(B, T, 1, device = coords.device)

        for t in range(T):

            mods = self.latent_to_mod(
                code,
                hs[:,t,:],
                static_emb[:, t, :] if static_emb is not None else None,
            )
            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            post_out     = self.layers[-1](pre_out)

            out[:,t] = post_out

        return out


    def _inr_batch_static_only(self, coords, code, x_statics, dir_idx):
        B, T, _ = coords.shape
        static_emb = self.static_encoder(x_statics, dir_idx)          # (B, S)
        position   = self.embedding(coords)                            # (B, T, pos_dim)
        out = torch.zeros(B, T, 1, device = coords.device)



        for t in range(T):

            mods = self.latent_to_mod(
                code,
                [None],
                static_emb[:,t,:],
            )

            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            post_out     = self.layers[-1](pre_out)

            out[:,t] = post_out

        return out

    def _inr_batch_vanilla(self, coords, code):

        B, T, _ = coords.shape
        position = self.embedding(coords)

        out = torch.zeros(B, T, 1, device = coords.device)
        mods          = self.latent_to_mod_vanilla(code)

        for t in range(T):

            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            out[:,t] = self.layers[-1](pre_out)

        return out


