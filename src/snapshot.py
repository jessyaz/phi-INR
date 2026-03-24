"""
src/snapshot.py
"""
import json, shutil, subprocess, sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

_SRC_FILES = [
    "src/network.py", "src/metalearning.py", "src/dataloaders.py",
    "src/head_sequencer.py", "src/film_conditionning.py", "inr_forecast.py",
]
_OPTIONAL_FILES = ["src/baseline_lstm.py", "src/snapshot.py"]


def save_snapshot(run_dir: Path, root: Path, cfg: DictConfig) -> None:
    """
    Appelée à chaque meilleur checkpoint dans train.py.
    Sauvegarde code source, config Hydra, schéma de données, métadonnées.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Code source
    for rel in _SRC_FILES + _OPTIONAL_FILES:
        src = root / rel
        if src.exists():
            dst = run_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Config Hydra
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Schéma de données (yaml + fingerprint JSON)
    schema_src = root / "conf" / "data_schema.yaml"
    if schema_src.exists():
        schema_dst = run_dir / "conf" / "data_schema.yaml"
        schema_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(schema_src, schema_dst)

    try:
        from src.dataloaders import schema_fingerprint  # type: ignore
        (run_dir / "data_schema_fingerprint.json").write_text(
            json.dumps(schema_fingerprint(), indent=2)
        )
    except Exception as e:
        print(f"  [Snapshot] ⚠ Fingerprint non sauvegardé : {e}")

    # Métadonnées git
    meta: dict = {"saved_at": datetime.now().isoformat()}
    try:
        meta["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        meta["git_hash"] = "unavailable"

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[Snapshot] ✓ {run_dir}")


@contextmanager
def frozen_src_context(snapshot_dir: Path):
    """
    Remplace temporairement src/ par la version gelée du snapshot.
    NZDataset reste chargé depuis le code actuel (dataset partagé).
    Utiliser check_schema_compat() avant pour valider la compatibilité.
    """
    snapshot_dir = Path(snapshot_dir)
    if not (snapshot_dir / "src").exists():
        raise FileNotFoundError(
            f"Pas de code gelé dans {snapshot_dir}/src/\n"
            f"  → Lance make_snapshot.py pour les anciens runs."
        )

    cached = {k: v for k, v in sys.modules.items()
              if k == "src" or k.startswith("src.")}
    for k in list(cached):
        del sys.modules[k]

    sys.path.insert(0, str(snapshot_dir))
    try:
        yield
    finally:
        sys.path.remove(str(snapshot_dir))
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(cached)


def check_schema_compat(snapshot_dir: Path) -> tuple[bool, list[str]]:
    """
    Vérifie que le schéma de données du snapshot est compatible avec le code actuel.
    Retourne (ok: bool, messages: list[str]).
    ok=False → les dimensions tenseurs ont changé, inférence incorrecte.
    """
    snapshot_dir = Path(snapshot_dir)
    fp_path = snapshot_dir / "data_schema_fingerprint.json"

    if not fp_path.exists():
        return True, ["(pas de fingerprint — ancien run, vérification ignorée)"]

    try:
        saved_fp = json.loads(fp_path.read_text())
    except Exception as e:
        return True, [f"(fingerprint illisible : {e})"]

    try:
        from src.dataloaders import check_compat  # type: ignore
        diffs = check_compat(saved_fp)
    except ImportError:
        return True, ["(dataloaders non importable)"]

    CRITICAL = {"meteo_dim", "x_time_dim", "lstm_in_dim",
                "col_meteo", "time_features", "use_cyclic_dow"}
    critical = [d for d in diffs if any(c in d for c in CRITICAL)]

    if critical:
        msgs = (
                ["⚠  Incompatibilité schéma de données :"]
                + [f"   {d}" for d in diffs]
                + ["   → Dimensions tenseurs modifiées — ce run ne peut pas être comparé."]
        )
        return False, msgs

    return True, [f"(diff non-critique) {d}" for d in diffs] if diffs else []


def list_snapshots(save_dir: Path) -> list[Path]:
    """Liste les snapshots (dossiers avec config.yaml + *.pt), plus récents en premier."""
    out = []
    for d in Path(save_dir).rglob("config.yaml"):
        folder = d.parent
        if list(folder.glob("*.pt")):
            out.append(folder)
    return sorted(set(out), key=lambda p: p.stat().st_mtime, reverse=True)


def snapshot_info(snapshot_dir: Path) -> dict:
    """Résumé lisible d'un snapshot."""
    snapshot_dir = Path(snapshot_dir)
    info: dict = {"path": str(snapshot_dir), "name": snapshot_dir.name}

    meta_path = snapshot_dir / "meta.json"
    if meta_path.exists():
        info.update(json.loads(meta_path.read_text()))

    ckpts = sorted(snapshot_dir.glob("*.pt"))
    info["checkpoint"] = str(ckpts[-1]) if ckpts else None
    info["frozen_src"] = (snapshot_dir / "src").exists()
    info["has_schema"] = (snapshot_dir / "data_schema_fingerprint.json").exists()

    cfg_path = snapshot_dir / "config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        info["look_back"]   = OmegaConf.select(cfg, "data.look_back_window", default="?")
        info["horizon"]     = OmegaConf.select(cfg, "data.horizon",          default="?")
        info["use_context"] = OmegaConf.select(cfg, "inr.use_context",       default="?")
        info["control"]     = OmegaConf.select(cfg, "inr.control",           default=None)
        info["latent_dim"]  = OmegaConf.select(cfg, "inr.latent_dim",        default="?")

    fp_path = snapshot_dir / "data_schema_fingerprint.json"
    if fp_path.exists():
        fp = json.loads(fp_path.read_text())
        info["lstm_in_dim"] = fp.get("lstm_in_dim", "?")
        info["x_time_dim"]  = fp.get("x_time_dim",  "?")
        info["time_feats"]  = fp.get("time_features","?")

    ok, diffs = check_schema_compat(snapshot_dir)
    info["schema_ok"]    = ok
    info["schema_diffs"] = diffs
    return info