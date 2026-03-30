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


METRIC_KEYS = ["mae", "rmse", "mape"]


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
    col_run  = 45
    col_val  = 13
    sep_len  = col_run + col_val * len(METRIC_KEYS)

    header = f"{'Run':<{col_run}}" + "".join(f"{k.upper():>{col_val}}" for k in METRIC_KEYS)
    print("\n" + "═" * sep_len)
    print("COMPARAISON")
    print("═" * sep_len)
    print(header)
    print("─" * sep_len)

    best: dict[str, tuple[str, float]] = {k: ("", float("inf")) for k in METRIC_KEYS}

    for run_name, res in results.items():
        short = run_name[-col_run:] if len(run_name) > col_run else run_name
        if "error" in res:
            print(f"{short:<{col_run}}  ✗  {res['error']}")
        else:
            vals = [res.get(k, float("nan")) for k in METRIC_KEYS]
            print(f"{short:<{col_run}}" + "".join(f"{v:>{col_val}.6f}" for v in vals))
            for k, v in zip(METRIC_KEYS, vals):
                if v < best[k][1]:
                    best[k] = (run_name, v)

    print("─" * sep_len)
    print("Meilleur par métrique :")
    for k, (name, val) in best.items():
        if name:
            short = name[-col_run:] if len(name) > col_run else name
            print(f"  {k.upper():>6} → {short}  ({val:.6f})")
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
            print("    " + "  ".join(f"{k.upper()}={metrics[k]:.6f}" for k in METRIC_KEYS))

    _print_table(results)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[✓] Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    main()