"""
Compare plusieurs snapshots sur un même jeu de test.

Usage:
    python compare_runs.py --parquet data/test.parquet \\
        --runs save_models/snapshots/exp1/run_A \\
               save_models/snapshots/exp1/run_B \\
               save_models/snapshots/exp2/run_C

    # Pour un parquet externe (hors splits_meta.json) :
    python compare_runs.py --parquet data/new_data.parquet --no-site-filter \\
        --runs save_models/snapshots/exp1/run_A ...
"""
import sys
import argparse
import json
import subprocess
from pathlib import Path


METRIC_KEYS = [
    "past_mae", "past_rmse", "past_mape", "past_smape",
    "horizon_mae", "horizon_rmse", "horizon_mape", "horizon_smape",
]


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation d'un snapshot dans un sous-process isolé
# ══════════════════════════════════════════════════════════════════════════════

def _eval_snapshot(
        snapshot_dir:   Path,
        parquet:        str,
        batch_size:     int,
        num_workers:    int,
        no_site_filter: bool = False,
) -> dict:
    eval_script = snapshot_dir / "evaluate.py"
    if not eval_script.exists():
        return {"error": f"evaluate.py introuvable dans {snapshot_dir}"}

    cmd = [
        sys.executable, str(eval_script),
        "--parquet",     parquet,
        "--batch-size",  str(batch_size),
        "--num-workers", str(num_workers),
        "--quiet",
    ]

    if no_site_filter:
        cmd.append("--no-site-filter")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        return {"error": err[-1] if err else "process error"}

    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("JSON:"):
            return json.loads(line[5:])

    return {"error": "Pas de ligne JSON: trouvée dans la sortie"}


# ══════════════════════════════════════════════════════════════════════════════
# Tableau comparatif
# ══════════════════════════════════════════════════════════════════════════════

def _print_table(results: dict[str, dict]) -> None:
    col_run = 35
    col_val = 11

    # Deux groupes : passé et horizon
    past_keys    = ["past_mae",    "past_rmse",    "past_mape",    "past_smape"]
    horizon_keys = ["horizon_mae", "horizon_rmse", "horizon_mape", "horizon_smape"]
    labels       = ["MAE", "RMSE", "MAPE", "SMAPE"]

    sep_len = col_run + col_val * len(labels) * 2 + 6

    print("\n" + "═" * sep_len)
    print("COMPARAISON")
    print("═" * sep_len)
    print(
        f"{'Run':<{col_run}}  "
        f"{'── PASSÉ (192h) ──':^{col_val * len(labels)}}  "
        f"{'── HORIZON (48h) ──':^{col_val * len(labels)}}"
    )
    print(
        f"{'':< {col_run}}  " +
        "".join(f"{l:>{col_val}}" for l in labels) + "  " +
        "".join(f"{l:>{col_val}}" for l in labels)
    )
    print("─" * sep_len)

    best = {k: ("", float("inf")) for k in past_keys + horizon_keys}

    for run_name, res in results.items():
        short = run_name[-col_run:] if len(run_name) > col_run else run_name
        if "error" in res:
            print(f"{short:<{col_run}}  ✗  {res['error']}")
        else:
            past_vals    = [res.get(k, float("nan")) for k in past_keys]
            horizon_vals = [res.get(k, float("nan")) for k in horizon_keys]
            row = f"{short:<{col_run}}  "
            row += "".join(f"{v:>{col_val}.4f}" for v in past_vals)
            row += "  "
            row += "".join(f"{v:>{col_val}.4f}" for v in horizon_vals)
            print(row)
            for k, v in zip(past_keys + horizon_keys, past_vals + horizon_vals):
                if v < best[k][1]:
                    best[k] = (run_name, v)

    print("─" * sep_len)
    print("Meilleur par métrique :")
    for k, (name, val) in best.items():
        if name:
            short = name[-col_run:] if len(run_name) > col_run else name
            print(f"  {k:>20} → {short}  ({val:.4f})")
    print("═" * sep_len)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare des snapshots figés")
    parser.add_argument("--parquet",        required=True,  help="Chemin vers le .parquet de test")
    parser.add_argument("--runs",           nargs="+", required=True, help="Dossiers snapshot à comparer")
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--num-workers",    type=int, default=0)
    parser.add_argument("--out",            default="comparison_results.json", help="Fichier JSON de sortie")
    parser.add_argument("--no-site-filter", action="store_true", help="Désactive le filtre splits_meta.json")
    args = parser.parse_args()

    past_keys    = ["past_mae",    "past_rmse",    "past_mape",    "past_smape"]
    horizon_keys = ["horizon_mae", "horizon_rmse", "horizon_mape", "horizon_smape"]

    results: dict[str, dict] = {}

    for run_path_str in args.runs:
        run_dir  = Path(run_path_str).resolve()
        run_name = run_dir.name
        print(f"\n[→] {run_name}")
        metrics = _eval_snapshot(
            run_dir, args.parquet, args.batch_size, args.num_workers,
            no_site_filter=args.no_site_filter,
        )
        results[run_name] = metrics
        if "error" in metrics:
            print(f"    ✗  {metrics['error']}")
        else:
            print("    PASSÉ   : " + "  ".join(f"{k.split('_')[1].upper()}={metrics[k]:.4f}" for k in past_keys if k in metrics))
            print("    HORIZON : " + "  ".join(f"{k.split('_')[1].upper()}={metrics[k]:.4f}" for k in horizon_keys if k in metrics))

    _print_table(results)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[✓] Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    main()