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

class StaticEncoder_LAST(nn.Module):
    """
    lat/lon → FourierSpatialEncoder → (B, spatial_dim)
    dir_idx → nn.Embedding          → (B, dir_dim)
    concat                          → (B, spatial_dim + dir_dim)
    """
    def __init__(self, spatial_dim=32, dir_dim=8, num_directions=6, sigma=0.1):
        super().__init__()
        self.spatial_enc   = FourierSpatialEncoder(num_freqs=16, d_out=spatial_dim, sigma=sigma)
        self.dir_embedding = nn.Embedding(num_embeddings=num_directions, embedding_dim=dir_dim)
        self.out_dim       = spatial_dim + dir_dim
     #   self.norm          = nn.LayerNorm(self.out_dim)

    def forward(self, x_statics, dir_idx):
        assert dir_idx.max() < self.dir_embedding.num_embeddings, (
            f"dir_idx max={dir_idx.max().item()} >= num_directions={self.dir_embedding.num_embeddings} "
            f"— augmente num_directions dans config.yaml"
        )
        spatial = self.spatial_enc(x_statics.float())
        dir_vec = self.dir_embedding(dir_idx)
        return torch.cat([spatial, dir_vec], dim=-1)#self.norm( torch.cat([spatial, dir_vec], dim=-1) )

class StaticEncoder(nn.Module):
    """
    lat/lon → MLP  → (B, spatial_dim)
    dir_idx → Embedding → (B, dir_dim)
    concat  → (B, spatial_dim + dir_dim)
    """
    def __init__(self, spatial_dim=32, dir_dim=8, num_directions=6, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, spatial_dim),
        )
        self.dir_embedding = nn.Embedding(num_embeddings=num_directions, embedding_dim=dir_dim)
        self.out_dim       = spatial_dim + dir_dim

    def forward(self, x_statics, dir_idx):
        assert dir_idx.max() < self.dir_embedding.num_embeddings, (
            f"dir_idx max={dir_idx.max().item()} >= num_directions={self.dir_embedding.num_embeddings}"
        )
        spatial = self.mlp(x_statics.float())
        dir_vec = self.dir_embedding(dir_idx)
        return torch.cat([spatial, dir_vec], dim=-1)

# ── Hypernetwork avec contexte ────────────────────────────────

class LatentToModulation(nn.Module):
    """code + h_dynamics + static_emb → modulations FiLM"""
    def __init__(self, latent_dim, lstm_hidden_dim, static_emb_dim, num_modulations, control = None):
        super().__init__()

        self.control = control


        if control == "static_only":
            print("HN Mode S : ", control)

            input_dim = latent_dim + static_emb_dim
            hidden_dim = input_dim * 2

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_modulations)
            )
        else:

            input_dim = latent_dim + static_emb_dim + lstm_hidden_dim
            hidden_dim = input_dim * 2

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_modulations)
            )

            print("HN Mode : ", control)

    def forward(self, code, h_t, static_emb):
        if self.control == "static_only":
            return self.net(torch.cat([code, static_emb], dim=-1))
        else:
            return self.net(torch.cat([code, h_t, static_emb], dim=-1))


class LatentToModulationCode(nn.Module):
    """code + h_dynamics + static_emb → modulations FiLM"""
    def __init__(self, latent_dim, static_emb_dim, num_modulations, control = None):
        super().__init__()

        self.control = control

        print("HN Mode SUPER : ", control)

        input_dim = latent_dim + static_emb_dim
        hidden_dim = input_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_modulations),
        )


# ── Hypernetwork vanilla ──────────────────────────────────────

class LatentToModulationVanilla(nn.Module):
    """code seul → modulations (INR vanilla)"""
    def __init__(self, latent_dim, num_modulations):
        super().__init__()
        self.net = nn.Linear(latent_dim, num_modulations)

    def forward(self, code):
        return self.net(code)


# ── Encodage positionnel NeRF ─────────────────────────────────

class NeRFEncoding(nn.Module):
    def __init__(self, num_frequencies, min_freq, include_input=True, input_dim=1, base_freq=1.25):
        super().__init__()
        self.include_input = include_input
        bands              = base_freq * torch.arange(min_freq, num_frequencies, dtype=torch.float32)
        #self.bands         = nn.Parameter(bands, requires_grad=False)
        self.register_buffer('bands', bands)

        self.out_dim       = bands.shape[0] * input_dim * 2

        print("CALSS", self.out_dim)
        if include_input:
            self.out_dim  += input_dim

    def forward(self, coords):
        winded  = coords[..., None, :] * self.bands[None, None, :, None]
        winded  = winded.reshape(*coords.shape[:2], -1)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
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
            include_input    = True,
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
            include_input   = include_input,
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
        )

        self.latent_to_mod_code = LatentToModulationCode(
            latent_dim      = latent_dim,
            static_emb_dim  = static_emb_dim,
            num_modulations = num_modulations,
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
        static_emb = self.static_encoder(x_statics, dir_idx)          # (B, S)
        position   = self.embedding(coords)                            # (B, T, pos_dim)
        out = torch.zeros(B, T, 1, device = coords.device)

        for t in range(T):

            mods = self.latent_to_mod(
                code,
                hs[:,t,:],
                static_emb,
            )
            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            post_out     = self.layers[-1](pre_out)

            out[:,t] = post_out

        return out


    def _inr_batch_static_only(self, coords, code, x_statics, dir_idx, special=None):
        B, T, _ = coords.shape
        static_emb = self.static_encoder(x_statics, dir_idx)          # (B, S)
        position   = self.embedding(coords)                            # (B, T, pos_dim)
        out = torch.zeros(B, T, 1, device = coords.device)

        if special == True:
            for t in range(T):

                mods_code = self.latent_to_mod_vanilla(
                    code
                )
                mods_statique =  self.latent_to_mod_code(
                    static_emb
                )

                pre_out = film_translate_spe(position[:,t].squeeze(-1), mods_code,mods_statique, self.layers[:-1], torch.relu)
                post_out     = self.layers[-1](pre_out)

                out[:,t] = post_out

            return out


        for t in range(T):

            mods = self.latent_to_mod(
                code,
                [None],
                static_emb,
            )
            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            post_out     = self.layers[-1](pre_out)

            out[:,t] = post_out

        return out

    def _inr_batch_vanilla(self, coords, code):

        B, T, _ = coords.shape
        position = self.embedding(coords)

        out = torch.zeros(B, T, 1, device = coords.device)

        for t in range(T):

            mods          = self.latent_to_mod_vanilla(code)
            pre_out = film_translate(position[:,t].squeeze(-1), mods, self.layers[:-1], torch.relu)
            out[:,t] = self.layers[-1](pre_out)

        return out


    def modulated_forward(self, coords_p, code, x_context_p, y_past,
                          x_statics=None, dir_idx=None,
                          coords_h=None, x_context_h=None, beta=0):
        B, T_p, _ = coords_p.shape


        print("P")

        #DEBUG??
        if not self.use_context:
            out_p = self._inr_batch_vanilla(coords_p, code)
            out_h = None
            if coords_h is not None:
                out_h = self._inr_batch_vanilla(coords_h, code)
            return out_p, out_h, torch.tensor(0.0, device=coords_p.device)

        # ── Contexte LSTM ─────────────────────────────────────────
        assert dir_idx   is not None, "dir_idx requis quand use_context=True"
        assert x_statics is not None, "x_statics requis quand use_context=True"

        if self._debug:
            print("  [context] LSTM forward_past...")
        hs_p, h_final, c_final = self.lstm.forward_past(x_context_p, y_past)

        if self._debug:
            print(f"  hs_p         : {tuple(hs_p.shape)}  nan={hs_p.isnan().any().item()}  norm={hs_p.norm():.4f}")
            print(f"  h_final      : {tuple(h_final.shape)}  nan={h_final.isnan().any().item()}")
            print("  [context] _inr_batch série P...")
        out_p = self._inr_batch(coords_p, code, hs_p, x_statics, dir_idx)

        if self._debug:
            print(f"  out_p        : {tuple(out_p.shape)}  nan={out_p.isnan().any().item()}  "
                  f"min={out_p.min():.4f} max={out_p.max():.4f}")

        out_h = None
        if coords_h is not None:
            if self._debug:
                print(f"  [context] série H AR — {coords_h.shape[1]} pas...")

            T_h       = coords_h.shape[1]
            h, c      = h_final, c_final
            last_flow = out_p[:, -1, :]
            hs_h      = []

            for t in range(T_h):
                h_since = torch.full((B, 1), (t + 1) / 24.0, device=coords_h.device)
                x_in    = torch.cat([x_context_h[:, t, :], last_flow, h_since], dim=-1)

                h, c = self.lstm.cell_step(x_in, h, c)

                if h.isnan().any():
                    print(f"    ⚠ NaN dans h à t={t}")
                    break

                hs_h.append(h.unsqueeze(1))

            hs_h = torch.cat(hs_h, dim=1)
            if self._debug:
                print(f"  hs_h         : {tuple(hs_h.shape)}  nan={hs_h.isnan().any().item()}  norm={hs_h.norm():.4f}")

            out_h = self._inr_batch(coords_h, code, hs_h, x_statics, dir_idx)
            if self._debug:
                print(f"  out_h        : {tuple(out_h.shape)}  nan={out_h.isnan().any().item()}  "
                  f"min={out_h.min():.4f} max={out_h.max():.4f}")
        if self._debug:
            print("[MFF.modulated_forward] ✓ done\n")
        return out_p, out_h, torch.tensor(0.0, device=coords_p.device)