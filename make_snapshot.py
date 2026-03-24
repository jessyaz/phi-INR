#!/usr/bin/env python
"""
make_snapshot.py — crée rétroactivement un snapshot pour un ancien .pt
"""
import argparse, json, shutil, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf, DictConfig

_SRC_FILES = [
    "src/network.py", "src/metalearning.py", "src/dataloaders.py",
    "src/head_sequencer.py", "src/film_conditionning.py",
    "src/snapshot.py", "inr_forecast.py",
]
_OPTIONAL = ["src/baseline_lstm.py"]


def _infer_cfg(ckpt: dict) -> DictConfig:
    ci = ckpt.get("cfg_inr",   {})
    cd = ckpt.get("cfg_data",  {})
    ck = ckpt.get("cfg_inner", {})
    return OmegaConf.create({
        "data": {
            "train_parquet": "data/train_data.parquet",
            "val_parquet":   "data/val_data.parquet",
            "test_parquet":  "data/test_data.parquet",
            "input_dim":        cd.get("input_dim",       1),
            "output_dim":       cd.get("output_dim",      1),
            "look_back_window": cd.get("look_back_window",192),
            "horizon":          cd.get("horizon",          48),
            "num_workers":      cd.get("num_workers",       1),
            "pin_memory":       cd.get("pin_memory",     True),
        },
        "inr": {
            "control":         ci.get("control",        None),
            "use_context":     ci.get("use_context",    True),
            "freeze_lstm":     ci.get("freeze_lstm",   False),
            "lstm_ckpt":       ci.get("lstm_ckpt",      None),
            "lstm_hidden_dim": ci.get("lstm_hidden_dim", 256),
            "latent_dim":      ci.get("latent_dim",      256),
            "hidden_dim":      ci.get("hidden_dim",      128),
            "depth":           ci.get("depth",             4),
            "num_frequencies": ci.get("num_frequencies",   8),
            "min_frequencies": ci.get("min_frequencies",   0),
            "base_frequency":  ci.get("base_frequency",  1.25),
            "include_input":   ci.get("include_input",  True),
            "static": {
                "spatial_dim":    ci.get("static",{}).get("spatial_dim",    256),
                "dir_dim":        ci.get("static",{}).get("dir_dim",         32),
                "num_directions": ci.get("static",{}).get("num_directions",   7),
                "sigma":          ci.get("static",{}).get("sigma",          0.1),
            },
        },
        "inner": {
            "inner_steps": ck.get("inner_steps", 3),
            "w_passed":    ck.get("w_passed",   1.0),
            "w_futur":     ck.get("w_futur",    1.0),
        },
    })


def make_snapshot(ckpt_path: Path, label: str | None = None,
                  hydra_cfg: Path | None = None) -> Path:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    snap_dir = ckpt_path.parent / (label or ckpt_path.stem)
    snap_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ {snap_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = OmegaConf.load(hydra_cfg) if hydra_cfg else _infer_cfg(ckpt)
    OmegaConf.save(cfg, snap_dir / "config.yaml")

    # .pt
    dst = snap_dir / ckpt_path.name
    if not dst.exists():
        shutil.copy2(ckpt_path, dst)

    # data_schema.yaml
    schema_src = ROOT / "conf" / "data_schema.yaml"
    if schema_src.exists():
        (snap_dir / "conf").mkdir(exist_ok=True)
        shutil.copy2(schema_src, snap_dir / "conf" / "data_schema.yaml")

    # fingerprint
    try:
        from src.dataloaders import schema_fingerprint
        (snap_dir / "data_schema_fingerprint.json").write_text(
            json.dumps(schema_fingerprint(), indent=2)
        )
        print(f"  fingerprint : lstm_in_dim={schema_fingerprint()['lstm_in_dim']} "
              f"x_time_dim={schema_fingerprint()['x_time_dim']}")
    except Exception as e:
        print(f"  ⚠ fingerprint : {e}")

    # code source
    for rel in _SRC_FILES + _OPTIONAL:
        src = ROOT / rel
        if src.exists():
            dst2 = snap_dir / rel
            dst2.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst2)

    meta = {
        "saved_at":       datetime.now().isoformat(),
        "source":         "make_snapshot.py (retroactive)",
        "original_ckpt":  str(ckpt_path),
        "git_hash":       "unknown",
        "warning":        "Code source = état actuel au moment de make_snapshot.py",
    }
    (snap_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    lb = OmegaConf.select(cfg, "data.look_back_window", default="?")
    h  = OmegaConf.select(cfg, "data.horizon",          default="?")
    c  = OmegaConf.select(cfg, "inr.control",           default="?")
    print(f"  ✓  look_back={lb} horizon={h} control={c}")
    return snap_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      nargs="+", required=True)
    p.add_argument("--label",     nargs="*", default=None)
    p.add_argument("--hydra_cfg", default=None,
                   help="conf/config.yaml — recommandé pour les runs récents")
    args = p.parse_args()

    labels    = args.label or [None] * len(args.ckpt)
    hydra_cfg = Path(args.hydra_cfg) if args.hydra_cfg else None

    created = []
    for ckpt_p, lbl in zip(args.ckpt, labels + [None]*(len(args.ckpt)-len(labels))):
        try:
            created.append(make_snapshot(Path(ckpt_p), lbl, hydra_cfg))
        except Exception as e:
            print(f"  ✗ {ckpt_p} : {e}")

    if created:
        print(f"\n{'─'*60}")
        print(f"Snapshots créés : {len(created)}")
        for s in created:
            print(f"  {s}")
        snaps = " \\\n    ".join(str(s) for s in created)
        print(f"\npython evaluate.py \\\n  --snapshots {snaps} \\\n"
              f"  --test_data data/test_data.parquet \\\n"
              f"  --scalers   save_models/inr/INR_LSTM_FREEZE/scalers.pkl \\\n"
              f"  --output    results/exp01")

if __name__ == "__main__":
    main()