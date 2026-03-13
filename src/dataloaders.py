from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# ── Colonnes ─────────────────────────────────────────────────

COL_METEO     = ['msl', 'tcc', 'u10', 'v10', 't2m', 'd2m', 'tp', 'cp', 'ssrd']
COL_STATIC    = ['LAT', 'LON']
COL_TARGET    = 'FLOW'
WEIGHT_TARGET = 0

# ── Scaler ───────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std  = None
        self.eps  = eps

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std  = x.std(axis=0,  keepdims=True)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

# ── Encodage cyclique ─────────────────────────────────────────

def encode_day_of_week(dow: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * dow / 7
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype('float32')

# ── Dataset ──────────────────────────────────────────────────

class NZDataset(Dataset):

    def __init__(self, parquet_file, mode='train', window='10D', scalers=None, latent_dim=128):
        self.mode       = mode
        self.latent_dim = latent_dim

        meta            = json.loads(Path('data/splits_meta.json').read_text())
        allowed_sites   = set(meta['common_sites'])


        self.window_h   = int(pd.to_timedelta(window).total_seconds() // 3600)

        df = pd.read_parquet(parquet_file)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df[df['SITEREF'].isin(allowed_sites)]
        df = df.sort_values(['SITEREF', 'DIRECTION', 'DATETIME']).reset_index(drop=True)

        # ── Scalers ──
        self.scalers = scalers
        if mode == 'train':
            self.scalers = {
                'meteo':  StandardScaler(),
                'target': StandardScaler(),
                'static': StandardScaler(),
            }
            df[COL_METEO] = self.scalers['meteo'].fit_transform(df[COL_METEO].values)
            mask = df[COL_TARGET].notna() & (df['WEIGHT'] == WEIGHT_TARGET)
            df.loc[mask, COL_TARGET] = self.scalers['target'].fit_transform(
                df.loc[mask, [COL_TARGET]].values
            )
            self.scalers['static'].fit(df.drop_duplicates('SITEREF')[COL_STATIC].values)
        else:
            if self.scalers is None:
                raise ValueError("scalers requis pour val/test")
            df[COL_METEO] = self.scalers['meteo'].transform(df[COL_METEO].values)
            mask = df[COL_TARGET].notna() & (df['WEIGHT'] == WEIGHT_TARGET)
            df.loc[mask, COL_TARGET] = self.scalers['target'].transform(
                df.loc[mask, [COL_TARGET]].values
            )

        # ── Encodage temporel cyclique ──
        df['dow_sin'] = encode_day_of_week(df['DATETIME'].dt.dayofweek.values)[:, 0]
        df['dow_cos'] = encode_day_of_week(df['DATETIME'].dt.dayofweek.values)[:, 1]

        # ── Construction des samples ──
        self.samples  = []
        n_series      = 0
        n_too_short   = 0
        n_ok          = 0

        for (site, direction), group in tqdm(
                df.groupby(['SITEREF', 'DIRECTION']),
                desc=f"[{mode}] Séries"
        ):
            # Statique avant filtre WEIGHT
            x_static = self.scalers['static'].transform(
                group[COL_STATIC].iloc[[0]].values
            ).squeeze(0).astype('float32')

            # Garde uniquement WEIGHT_TARGET, trie par temps
            serie = group[group['WEIGHT'] == WEIGHT_TARGET].sort_values('DATETIME')
            if serie.empty:
                continue

            n_series += 1
            series_len = len(serie)

            if series_len < self.window_h:
                n_too_short += 1
                continue

            # Arrays numpy de la série complète
            arr_y     = serie[COL_TARGET].values.astype('float32')
            arr_meteo = serie[COL_METEO].values.astype('float32')
            arr_time  = np.stack([
                serie['dow_sin'].values,
                serie['dow_cos'].values,
                serie['IS_HOLIDAY'].values,
            ], axis=-1).astype('float32')

            # Découpe en fenêtres non-chevauchantes alignées sur la série
            t = np.linspace(-1, 1, self.window_h, dtype='float32').reshape(-1, 1)

            for start in range(0, series_len - self.window_h + 1, self.window_h):
                end = start + self.window_h
                self.samples.append({
                    't':        t,
                    'dir_idx':  np.array(direction, dtype=np.int64),
                    'x_static': x_static,
                    'x_meteo':  arr_meteo[start:end],
                    'x_time':   arr_time [start:end],
                    'y':        arr_y    [start:end].reshape(-1, 1),
                })
                n_ok += 1

        print(f"\nSéries total    : {n_series}")
        print(f"Séries trop courtes : {n_too_short}")
        print(f"Fenêtres générées   : {n_ok}")

        self.z = torch.zeros(len(self.samples), self.latent_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s['t']),
            torch.tensor(s['dir_idx']),
            torch.from_numpy(s['x_static']),
            torch.from_numpy(s['x_meteo']),
            torch.from_numpy(s['x_time']),
            torch.from_numpy(s['y']),
            self.z[idx],
        )

    def __setitem__(self, idx, z_values):
        self.z[idx] = z_values.clone()


    @staticmethod
    def check_splits(train_path: str, val_path: str, test_path: str, window: str = '10D'):
        window_h = int(pd.to_timedelta(window).total_seconds() // 3600)

        dfs = {}
        for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
            df = pd.read_parquet(path, columns=['SITEREF', 'DIRECTION', 'DATETIME', 'WEIGHT'])
            dfs[name] = df[df['WEIGHT'] == 0].copy()
            dfs[name]['DATETIME'] = pd.to_datetime(dfs[name]['DATETIME'])

        site_sets = {name: set(df['SITEREF'].unique()) for name, df in dfs.items()}
        common    = site_sets['train'] & site_sets['val'] & site_sets['test']

        for name, df in dfs.items():
            stats = df[df['SITEREF'].isin(common)].groupby(['SITEREF', 'DIRECTION'])['DATETIME'].count()
            print(f"{name.upper():<6} | sites: {len(site_sets[name])} | fenêtres: {(stats // window_h).sum()} | trop courtes: {(stats < window_h).sum()}")

        print(f"\nSites communs : {len(common)} / train={len(site_sets['train'])} val={len(site_sets['val'])} test={len(site_sets['test'])}")

        meta = {'common_sites': sorted(common), 'window': window}
        Path('data/splits_meta.json').write_text(json.dumps(meta, indent=2))

        return common


# ── Debug ────────────────────────────────────────────────────

if __name__ == "__main__":

    NZDataset.check_splits(
        'data/train_data.parquet',
        'data/val_data.parquet',
        'data/test_data.parquet',
        window='10D',
    )

    trainset = NZDataset('data/train_data.parquet', mode='train')
    valset   = NZDataset('data/val_data.parquet',   mode='val',  scalers=trainset.scalers)
    testset  = NZDataset('data/test_data.parquet',  mode='test', scalers=trainset.scalers)

    loader = DataLoader(trainset, batch_size=32, shuffle=True)
    t, dir_idx, x_static, x_meteo, x_time, y, z = next(iter(loader))

    print(f"t        : {t.shape}")          # (B, T, 1)
    print(f"dir_idx  : {dir_idx.shape}")    # (B,)
    print(f"x_static : {x_static.shape}")   # (B, 2)
    print(f"x_meteo  : {x_meteo.shape}")    # (B, T, 9)
    print(f"x_time   : {x_time.shape}")     # (B, T, 3)
    print(f"y        : {y.shape}")          # (B, T, 1)
    print(f"z        : {z.shape}")          # (B, 128)

    print(f"\nSamples — train: {len(trainset)}  val: {len(valset)}  test: {len(testset)}")