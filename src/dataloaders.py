from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# ══════════════════════════════════════════════════════════════════════════════
# Chargement du schéma centralisé
# ══════════════════════════════════════════════════════════════════════════════

def _load_schema(schema_path: Path | None = None) -> dict:
    """
    Charge conf/data_schema.yaml.
    Cherche dans l'ordre :
      1. schema_path explicite
      2. <racine_projet>/conf/data_schema.yaml   (src/../conf/)
      3. conf/data_schema.yaml depuis le CWD
    """
    candidates = []
    if schema_path:
        candidates.append(Path(schema_path))
    _here = Path(__file__).resolve().parent          # src/
    candidates.append(_here.parent / "conf" / "data_schema.yaml")
    candidates.append(Path("conf") / "data_schema.yaml")

    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f)
            return raw["data_schema"]

    raise FileNotFoundError(
        "conf/data_schema.yaml introuvable.\n"
        f"  Cherché dans : {[str(c) for c in candidates]}"
    )


_SCHEMA = _load_schema()

# ══════════════════════════════════════════════════════════════════════════════
# Constantes publiques dérivées du schéma
# ══════════════════════════════════════════════════════════════════════════════

COL_METEO     : list[str] = _SCHEMA["col_meteo"]
COL_STATIC    : list[str] = _SCHEMA["col_static"]
COL_TARGET    : str       = _SCHEMA["col_target"]
WEIGHT_TARGET : int       = _SCHEMA["weight_target"]

_USE_CYCLIC   : bool      = _SCHEMA.get("use_cyclic_dow", False)
_TIME_FEATS   : list[str] = _SCHEMA["time_features"]

METEO_DIM     : int = len(COL_METEO)
STAT_DIM      : int = len(COL_STATIC)

def _compute_x_time_dim() -> int:
    dim = 0
    for feat in _TIME_FEATS:
        dim += 2 if (feat == "dow" and _USE_CYCLIC) else 1
    return dim

X_TIME_DIM  : int = _compute_x_time_dim()
LSTM_IN_DIM : int = METEO_DIM

print(
    f"[DataSchema] meteo={METEO_DIM}  x_time={X_TIME_DIM}  "
    f"lstm_in={LSTM_IN_DIM}  static={STAT_DIM}  "
    f"feats={_TIME_FEATS}  cyclic={_USE_CYCLIC}"
)

# ══════════════════════════════════════════════════════════════════════════════
# Export / vérification du schéma (utilisé par snapshot.py et evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

def schema_fingerprint() -> dict:
    """Constantes critiques — sauvegardé dans chaque snapshot."""
    return {
        "col_meteo":      COL_METEO,
        "col_static":     COL_STATIC,
        "col_target":     COL_TARGET,
        "weight_target":  WEIGHT_TARGET,
        "time_features":  _TIME_FEATS,
        "use_cyclic_dow": _USE_CYCLIC,
        "meteo_dim":      METEO_DIM,
        "x_time_dim":     X_TIME_DIM,
        "lstm_in_dim":    LSTM_IN_DIM,
    }


def check_compat(saved_fp: dict) -> list[str]:
    """
    Compare un fingerprint sauvegardé avec les constantes actuelles.
    Retourne la liste des divergences (vide = compatible).
    """
    current = schema_fingerprint()
    return [
        f"{k}: snapshot={saved_fp.get(k)!r}  actuel={cur!r}"
        for k, cur in current.items()
        if saved_fp.get(k) != cur
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Scaler
# ══════════════════════════════════════════════════════════════════════════════

class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std  = None
        self.eps  = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean = x.mean(axis=0, keepdims=True)
        self.std  = x.std(axis=0,  keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ══════════════════════════════════════════════════════════════════════════════
# Encodage temporel
# ══════════════════════════════════════════════════════════════════════════════

def _build_time_features(serie: pd.DataFrame) -> np.ndarray:
    """
    Construit arr_time (N, X_TIME_DIM).
    Piloté entièrement par data_schema.yaml → time_features.
    Pour ajouter une feature : l'ajouter dans le yaml + gérer le cas ici.
    """
    cols = []
    for feat in _TIME_FEATS:
        if feat == "dow":
            if _USE_CYCLIC:
                angle = 2 * np.pi * serie["DATETIME"].dt.dayofweek.values / 7
                cols.append(np.sin(angle).astype("float32"))
                cols.append(np.cos(angle).astype("float32"))
            else:
                cols.append(serie["DATETIME"].dt.dayofweek.values.astype("float32"))

        elif feat == "is_holiday":

            cols.append(serie["IS_HOLIDAY"].values.astype("float32"))
        else:
            raise ValueError(f"Feature temporelle inconnue dans data_schema.yaml : {feat!r}")

    return np.stack(cols, axis=-1)   # (N, X_TIME_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class NZDataset(Dataset):

    def __init__(
            self,
            parquet_file,
            mode:       str = "train",
            window:     str = "10D",
            scalers         = None,
            latent_dim: int = 128,
            skip_site_filter = False
    ):
        self.mode       = mode
        self.latent_dim = latent_dim

        meta          = json.loads(Path("data/splits_meta.json").read_text())
        allowed_sites = set(meta["common_sites"])
        self.window_h = int(pd.to_timedelta(window).total_seconds() // 3600)

        df = pd.read_parquet(parquet_file)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])

        if not skip_site_filter:
            print("Skipping filter")
            df = df[df["SITEREF"].isin(allowed_sites)]
        else:
            print("apply filter")

        df = df.sort_values(["SITEREF", "DIRECTION", "DATETIME"]).reset_index(drop=True)

        # ── Scalers ──
        self.scalers = scalers
        if mode == "train":
            self.scalers = {
                "meteo":  StandardScaler(),
                "target": StandardScaler(),
                "static": StandardScaler(),
            }
            df[COL_METEO] = self.scalers["meteo"].fit_transform(df[COL_METEO].values)
            mask = df[COL_TARGET].notna() & (df["WEIGHT"] == WEIGHT_TARGET)
            df.loc[mask, COL_TARGET] = self.scalers["target"].fit_transform(
                df.loc[mask, [COL_TARGET]].values
            )
            self.scalers["static"].fit(
                df.drop_duplicates("SITEREF")[COL_STATIC].values
            )
        else:
            if self.scalers is None:
                raise ValueError("scalers requis pour val/test")
            df[COL_METEO] = self.scalers["meteo"].transform(df[COL_METEO].values)
            mask = df[COL_TARGET].notna() & (df["WEIGHT"] == WEIGHT_TARGET)
            df.loc[mask, COL_TARGET] = self.scalers["target"].transform(
                df.loc[mask, [COL_TARGET]].values
            )

        # ── Construction des samples ──
        self.samples: list[dict] = []
        n_series = n_too_short = n_ok = 0

        for (site, direction), group in tqdm(
                df.groupby(["SITEREF", "DIRECTION"]),
                desc=f"[{mode}] Séries",
        ):


            serie = group[group["WEIGHT"] == WEIGHT_TARGET].sort_values("DATETIME")
            if serie.empty:
                continue

            n_series   += 1
            series_len  = len(serie)

            if series_len < self.window_h:
                n_too_short += 1
                continue

            arr_y     = serie[COL_TARGET].values.astype("float32")
            arr_meteo = serie[COL_METEO].values.astype("float32")

            arr_static = self.scalers["static"].transform(
                serie[COL_STATIC].iloc[[0]].values # Uniquement lon et lat
            ).astype("float32").squeeze(0)

            arr_time  = _build_time_features(serie)        # (N, X_TIME_DIM)
            t         = np.linspace(-1, 1, self.window_h, dtype="float32").reshape(-1, 1)

            for start in range(0, series_len - self.window_h + 1, self.window_h):
                end = start + self.window_h
                self.samples.append({
                    "t":        t,
                    "dir_idx":  np.array(direction, dtype=np.int64),
                    "x_static": arr_static,
                    "x_meteo":  arr_meteo[start:end],
                    "x_time":   arr_time [start:end],
                    "y":        arr_y    [start:end].reshape(-1, 1),
                })
                n_ok += 1

        print(f"\nSéries total        : {n_series}")
        print(f"Séries trop courtes : {n_too_short}")
        print(f"Fenêtres générées   : {n_ok}")

        self.z = torch.zeros(len(self.samples), self.latent_dim)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["t"]),
            torch.tensor(s["dir_idx"]),
            torch.from_numpy(s["x_static"]),
            torch.from_numpy(s["x_meteo"]),
            torch.from_numpy(s["x_time"]),
            torch.from_numpy(s["y"]),
            self.z[idx],
        )

    def __setitem__(self, idx: int, z_values: torch.Tensor):
        self.z[idx] = z_values.clone()

    @staticmethod
    def check_splits(train_path, val_path, test_path, window="10D") -> set:
        window_h = int(pd.to_timedelta(window).total_seconds() // 3600)
        dfs = {}
        for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            df = pd.read_parquet(path, columns=["SITEREF","DIRECTION","DATETIME","WEIGHT"])
            dfs[name] = df[df["WEIGHT"] == 0].copy()
            dfs[name]["DATETIME"] = pd.to_datetime(dfs[name]["DATETIME"])

        site_sets = {n: set(df["SITEREF"].unique()) for n, df in dfs.items()}
        common    = site_sets["train"] & site_sets["val"] & site_sets["test"]

        for name, df in dfs.items():
            stats = (df[df["SITEREF"].isin(common)]
                     .groupby(["SITEREF","DIRECTION"])["DATETIME"].count())
            print(f"{name.upper():<6} | sites:{len(site_sets[name])} "
                  f"| fenêtres:{(stats//window_h).sum()} "
                  f"| trop courtes:{(stats<window_h).sum()}")

        print(f"\nSites communs : {len(common)} / "
              f"train={len(site_sets['train'])} val={len(site_sets['val'])} test={len(site_sets['test'])}")

        meta = {"common_sites": sorted(common), "window": window}
        Path("data/splits_meta.json").write_text(json.dumps(meta, indent=2))
        return common


# ── Debug ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    print("\n── Schéma actuel ──────────────────────────────")
    pprint.pprint(schema_fingerprint())

    NZDataset.check_splits(
        "data/train_data.parquet",
        "data/val_data.parquet",
        "data/test_data.parquet",
        window="10D",
    )

    trainset = NZDataset("data/train_data.parquet", mode="train")
    valset   = NZDataset("data/val_data.parquet",   mode="val",  scalers=trainset.scalers)
    testset  = NZDataset("data/test_data.parquet",  mode="test", scalers=trainset.scalers)

    loader = DataLoader(trainset, batch_size=32, shuffle=True)
    t, dir_idx, x_static, x_meteo, x_time, y, z = next(iter(loader))

    print(f"\nt        : {t.shape}")        # (B, T, 1)
    print(f"dir_idx  : {dir_idx.shape}")    # (B,)
    print(f"x_static : {x_static.shape}")   # (B, 2)
    print(f"x_meteo  : {x_meteo.shape}")    # (B, T, 9)
    print(f"x_time   : {x_time.shape}")     # (B, T, X_TIME_DIM)
    print(f"y        : {y.shape}")          # (B, T, 1)
    print(f"z        : {z.shape}")
    print(f"\nSamples — train:{len(trainset)}  val:{len(valset)}  test:{len(testset)}")