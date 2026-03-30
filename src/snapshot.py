"""
Snapshot : freeze du code source + configs dans run_dir/code/.
L'évaluateur autonome (evaluate.py) est copié dans run_dir/.
"""
import shutil
from pathlib import Path
from datetime import datetime

_EVAL_TEMPLATE = Path(__file__).parent / "_eval_template.py"


def save_snapshot(run_dir: Path, root: Path, cfg=None) -> None:
    """
    Sauvegarde dans run_dir/ :
      code/src/   ← src/ figé au moment du run
      code/conf/  ← conf/ figé au moment du run
      evaluate.py ← évaluateur autonome (copié depuis _eval_template.py)
      README.md   ← infos de run
    """
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # ── Source Python ─────────────────────────────────────────────────────────
    src_dst = code_dir / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(root / "src", src_dst)

    # ── Configs YAML ──────────────────────────────────────────────────────────
    conf_dst = code_dir / "conf"
    if conf_dst.exists():
        shutil.rmtree(conf_dst)
    shutil.copytree(root / "conf", conf_dst)

    # ── Évaluateur autonome ───────────────────────────────────────────────────
    shutil.copy2(_EVAL_TEMPLATE, run_dir / "evaluate.py")

    # ── README ────────────────────────────────────────────────────────────────
    (run_dir / "README.md").write_text(
        f"# Snapshot — {run_dir.name}\n\n"
        f"Généré le : {datetime.now().isoformat()}\n\n"
        f"## Évaluation\n\n"
        f"```bash\n"
        f"python evaluate.py --parquet /chemin/vers/test.parquet\n"
        f"```\n\n"
        f"## Comparaison multi-runs\n\n"
        f"```bash\n"
        f"python compare_runs.py --parquet /chemin/vers/test.parquet \\\\\n"
        f"    --runs {run_dir} <autre_run_dir>\n"
        f"```\n"
    )

    print(f"[Snapshot] code figé → {code_dir}")
    print(f"[Snapshot] évaluateur → {run_dir / 'evaluate.py'}")