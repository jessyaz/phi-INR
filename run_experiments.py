"""
Lance les expériences INR en parallèle (2 à la fois).

Expériences :
  1. vanilla              — use_context=False
  2. context_scratch      — use_context=True,  lstm from scratch
  3. context_loaded_free  — use_context=True,  lstm chargé, poids libres
  4. context_loaded_frozen— use_context=True,  lstm chargé, poids gelés

Usage :
  python run_experiments.py
  python run_experiments.py --lstm_ckpt save_models/head_lstm/intrigued-fly-607_20260313_1514_best.pt
  python run_experiments.py --lstm_ckpt save_models/head_lstm/intrigued-fly-607_20260313_1514_best.pt --workers 2
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from sympy.codegen.ast import none

ROOT = Path(__file__).resolve().parent


# ── Définition des expériences ────────────────────────────────

def make_experiments(lstm_ckpt: str | None) -> list[dict]:
    base = dict(
        epochs     = 300,
        batch_size = 32,
        lr_inr     = 1e-4,
        lr_code    = 1e-3,
    )

    exps = [
        {
            **base,
            'name':        'INR_STATIC',
            'use_context': True,
            'control': "static_only",
            'lstm_ckpt':   None, #"save_models/head_lstm/debonair-wolf-926_h256_20260316_1006_best.pt",
            'freeze_lstm': False,
            'latent_dim' :  512,
            'hidden_dim' :  128,  # Width
            'depth' :  4,
            'description': 'Recherche INR W STATIC',
        },
        {
            **base,
            'name':        'INR_STATIC',
            'use_context': True,
            'control': None,
            'lstm_ckpt':   None, #"save_models/head_lstm/debonair-wolf-926_h256_20260316_1006_best.pt",
            'freeze_lstm': False,
            'latent_dim' :  512,
            'hidden_dim' :  128,  # Width
            'depth' :  4,
            'description': 'Recherche INR W STATIC',
        },
        {
            **base,
            'name':        'INR_STATIC',
            'use_context': False,
            'control': None,
            'lstm_ckpt':  None,
            'freeze_lstm': None,
            'latent_dim' :  512,
            'hidden_dim' :  128,  # Width
            'depth' :  4,
            'description': 'Recherche INR W STATIC',
        }

    ]


    #{
   #     **base,
   #     'name':        'INR_CODE_RESEARCH',
   #     'use_context': False,
   #     #'lstm_ckpt':   lstm_ckpt,
   #     'freeze_lstm': False,
   #     'description': 'Contexte LSTM chargé — poids libres',
   # },
    return exps


# ── Lancement d'une expérience ────────────────────────────────

def run_experiment(exp: dict) -> dict:
    name     = exp['name']
    save_dir = ROOT / 'save_models' / 'inr' / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'train.log'

    overrides = [
        f"meta.experiment_pack={name}",
        f"paths.save_models=save_models/inr/{name}",
        f"paths.scalers_file=save_models/inr/{name}/scalers.pkl",

        f"inr.latent_dim={exp['latent_dim']}",
        f"inr.hidden_dim={exp['hidden_dim']}",
        f"inr.depth={exp['depth']}",


        f"inr.control= {str(exp['control'])}",
        f"inr.use_context={str(exp['use_context']).lower()}",
        f"inr.freeze_lstm={str(exp['freeze_lstm']).lower()}",

        f"inr.lstm_ckpt={'null' if exp['lstm_ckpt'] is None else exp['lstm_ckpt']}",
        f"optim.epochs={exp['epochs']}",
        f"optim.batch_size={exp['batch_size']}",
        f"optim.lr_inr={exp['lr_inr']}",
        f"optim.lr_code={exp['lr_code']}",
        f"hydra.run.dir=outputs/{name}",
    ]

    cmd   = [sys.executable, 'inr_forecast.py'] + overrides
    start = datetime.now()

    print(f"\n[START] {name}")
    print(f"  {exp['description']}")
    print(f"  log → {log_path}")

    with open(log_path, 'w') as log_f:
        log_f.write(f"Expérience : {name}\n")
        log_f.write(f"Description: {exp['description']}\n")
        log_f.write(f"Start      : {start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Command    : {' '.join(cmd)}\n")
        log_f.write("─" * 60 + "\n\n")
        log_f.flush()

        proc = subprocess.run(
            cmd,
            cwd    = ROOT,
            stdout = log_f,
            stderr = subprocess.STDOUT,
            text   = True,
        )

    end     = datetime.now()
    elapsed = end - start
    status  = 'OK' if proc.returncode == 0 else f'ERREUR (code {proc.returncode})'

    with open(log_path, 'a') as log_f:
        log_f.write(f"\n{'─' * 60}\n")
        log_f.write(f"End    : {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Durée  : {elapsed}\n")
        log_f.write(f"Status : {status}\n")

    print(f"[{status}] {name}  ({elapsed})")

    return {
        'name':        name,
        'status':      status,
        'returncode':  proc.returncode,
        'elapsed':     str(elapsed),
        'log':         str(log_path),
        'description': exp['description'],
    }


# ── Résumé ────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    print("\n" + "═" * 60)
    print("RÉSUMÉ DES EXPÉRIENCES")
    print("═" * 60)
    for r in results:
        icon = "✓" if r['returncode'] == 0 else "✗"
        print(f"  [{icon}] {r['name']:<30}  {r['elapsed']}")
        print(f"       {r['description']}")
        print(f"       log : {r['log']}")
    print("═" * 60)

    failed = [r for r in results if r['returncode'] != 0]
    if failed:
        print(f"\n⚠  {len(failed)} expérience(s) en erreur :")
        for r in failed:
            print(f"   - {r['name']} → {r['log']}")
    else:
        print("\n✓  Toutes les expériences terminées avec succès.")


def save_summary(results: list[dict], lstm_ckpt: str | None):
    summary_path = ROOT / 'save_models' / 'inr' / 'experiments_summary.txt'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(f"Expériences lancées le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"lstm_ckpt : {lstm_ckpt or 'non fourni'}\n\n")
        for r in results:
            icon = "✓" if r['returncode'] == 0 else "✗"
            f.write(f"[{icon}] {r['name']}\n")
            f.write(f"    {r['description']}\n")
            f.write(f"    status  : {r['status']}\n")
            f.write(f"    elapsed : {r['elapsed']}\n")
            f.write(f"    log     : {r['log']}\n\n")
    print(f"\nRésumé → {summary_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_ckpt', type=str, default=None,
                        help="Chemin vers le checkpoint head_lstm (optionnel)")
    parser.add_argument('--workers', type=int, default=2,
                        help="Nombre d'expériences en parallèle (défaut: 2)")
    args = parser.parse_args()

    exps = make_experiments(args.lstm_ckpt)

    print("═" * 60)
    print(f"LANCEMENT DE {len(exps)} EXPÉRIENCES ({args.workers} en parallèle)")
    print("═" * 60)
    for i, e in enumerate(exps):
        ckpt_info = Path(e['lstm_ckpt']).name if e['lstm_ckpt'] else 'scratch'
        print(f"  {i+1}. {e['name']:<30}  {ckpt_info}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in exps}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                exp = futures[future]
                print(f"[EXCEPTION] {exp['name']} : {e}")
                results.append({
                    'name':        exp['name'],
                    'status':      f'EXCEPTION: {e}',
                    'returncode':  -1,
                    'elapsed':     'N/A',
                    'log':         'N/A',
                    'description': exp['description'],
                })

    order = {e['name']: i for i, e in enumerate(exps)}
    results.sort(key=lambda r: order.get(r['name'], 99))

    print_summary(results)
    save_summary(results, args.lstm_ckpt)


if __name__ == '__main__':
    main()