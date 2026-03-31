"""
evaluate_imputation.py
======================
Évalue la robustesse des snapshots face aux données manquantes.

Usage:
    python evaluate_imputation.py \\
        --parquet data/test_data.parquet \\
        --no-site-filter \\
        --missing-rates 0.1 0.2 0.3 0.5 \\
        --runs save_models/snapshots/ablation_v2/full \\
               save_models/snapshots/ablation_v2/vanilla \\
        --out results/imputation_results.json
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path

METRIC_KEYS = ["mae", "rmse", "mape", "smape"]


def _eval_with_missing(
        snapshot_dir:   Path,
        parquet:        str,
        missing_rate:   float,
        batch_size:     int,
        num_workers:    int,
        no_site_filter: bool,
        seed:           int = 42,
) -> dict:
    worker = Path(__file__).parent / "_imputation_worker.py"
    if not worker.exists():
        return {"error": f"_imputation_worker.py introuvable dans {Path(__file__).parent}"}

    cmd = [
        sys.executable, str(worker),
        "--snapshot",      str(snapshot_dir),
        "--parquet",       parquet,
        "--missing-rate",  str(missing_rate),
        "--batch-size",    str(batch_size),
        "--num-workers",   str(num_workers),
        "--seed",          str(seed),
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

    return {"error": "Pas de ligne JSON trouvée"}


def _print_table(results: dict, missing_rates: list) -> None:
    col_run  = 35
    col_met  = 9
    rates_str = [f"{int(r*100)}%" for r in missing_rates]

    print("\n" + "═" * 120)
    print("ÉVALUATION IMPUTATION")
    print("═" * 120)

    # Header
    header = f"{'Run':<{col_run}}"
    for r in rates_str:
        header += f"  [{r:>4}] " + " ".join(f"{k.upper():>{col_met}}" for k in METRIC_KEYS)
    print(header)
    print("─" * 120)

    for run_name, by_rate in results.items():
        short = run_name[-col_run:] if len(run_name) > col_run else run_name
        row = f"{short:<{col_run}}"
        for r in missing_rates:
            res = by_rate.get(str(r), {})
            if "error" in res:
                row += f"  [ERR ] " + " ".join(f"{'--':>{col_met}}" for _ in METRIC_KEYS)
            else:
                row += f"  [{rates_str[missing_rates.index(r)]:>4}] "
                row += " ".join(f"{res.get(k, float('nan')):>{col_met}.4f}" for k in METRIC_KEYS)
        print(row)

    print("═" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",        required=True)
    parser.add_argument("--runs",           nargs="+", required=True)
    parser.add_argument("--missing-rates",  nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--num-workers",    type=int, default=0)
    parser.add_argument("--no-site-filter", action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out",            default="results/imputation_results.json")
    args = parser.parse_args()

    results = {}

    for run_path_str in args.runs:
        run_dir  = Path(run_path_str).resolve()
        run_name = run_dir.name
        results[run_name] = {}
        print(f"\n[→] {run_name}")

        for rate in args.missing_rates:
            print(f"    missing={int(rate*100)}%  ", end="", flush=True)
            metrics = _eval_with_missing(
                snapshot_dir   = run_dir,
                parquet        = args.parquet,
                missing_rate   = rate,
                batch_size     = args.batch_size,
                num_workers    = args.num_workers,
                no_site_filter = args.no_site_filter,
                seed           = args.seed,
            )
            results[run_name][str(rate)] = metrics
            if "error" in metrics:
                print(f"✗  {metrics['error']}")
            else:
                print("  ".join(f"{k.upper()}={metrics[k]:.4f}" for k in METRIC_KEYS))

    _print_table(results, args.missing_rates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[✓] Résultats → {out_path}")


if __name__ == "__main__":
    main()