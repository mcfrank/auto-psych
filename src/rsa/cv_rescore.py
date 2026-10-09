"""Re-score a finished RSA loop cell's models with grouped cross-validation.

    uv run python -m src.rsa.cv_rescore --cell data/rsa/sherlock_run1/real_rep1 \
        --train <the cell's training CSV> --work <scratch dir>

For cells that ran before the loop selected on grouped CV (Sherlock run 1):
every model the cell brought back (live, pruned) and every starting model is
scored by grouped CV on the cell's training trials (src.rsa.loop.cv), and the
per-model, per-source totals are written to ``<cell>/cv_rescore.json``
(aggregates only: no trial-level values). The training CSV is not in the
repository (trial-level data); ``--train`` must hash as the cell recorded it
(``data.sha256``), or the command refuses. Fits go to ``<work>/cache``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import tyro

from src.rsa.dataset import load_forced_choice
from src.rsa.fit import FitSettings
from src.rsa.loop.cv import DEFAULT_FOLDS, CVResult, compare_cv, cv_pointwise, make_folds, source_labels
from src.rsa.loop.fitting import ModelFailure
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@dataclass
class Args:
    cell: Path
    """A brought-back cell (models/, models/pruned/, data.sha256)."""
    train: Path
    """The cell's training trials (its responses.csv / train.csv)."""
    work: Path
    """Scratch directory for folds and fit cache."""
    folds: int = DEFAULT_FOLDS
    fold_seed: int = 0
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    workers: int = 5
    time_limit_sec: float = 3600


def _check_train(cell: Path, train: Path) -> str:
    sha = hashlib.sha256(Path(train).read_bytes()).hexdigest()
    record = cell / "data.sha256"
    if record.exists() and sha not in record.read_text():
        raise ValueError(f"{train} (sha256 {sha[:12]}) is not the training data {record} records")
    return sha


def model_files(cell: Path) -> Dict[str, Path]:
    files = {p.stem: p for p in sorted((cell / "models" / "pruned").glob("*.py"))}
    files.update({p.stem: p for p in sorted((cell / "models").glob("*.py"))})
    from src.models.model_manifest import read_manifest_names

    for name in read_manifest_names(SEEDS):
        files.setdefault(name, SEEDS / f"{name}.py")
    return files


def main(args: Args) -> dict:
    cell, work = Path(args.cell), Path(args.work)
    sha = _check_train(cell, args.train)
    folds = make_folds(args.train, work / "folds", args.folds, seed=args.fold_seed)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fold_seed)
    results: Dict[str, CVResult] = {}
    failed = {}
    for name, path in model_files(cell).items():
        try:
            results[name] = cv_pointwise(path, name, args.train, folds, settings, work / "cache",
                                         time_limit_sec=args.time_limit_sec, workers=args.workers)
            print(f"  {name}: ELPD-CV {results[name].elpd:.1f}", flush=True)
        except ModelFailure as exc:
            failed[name] = str(exc)
            print(f"  {name}: failed ({exc})", flush=True)
    sources = source_labels(load_forced_choice(args.train).frame)
    table = compare_cv(results, folds.units, sources)
    out = dict(
        cell=cell.name, train_sha256=sha, folds=args.folds, fold_seed=args.fold_seed,
        settings=vars(settings), n_trials=len(folds.fold), failed=failed,
        models={n: {k: v for k, v in row.items()} for n, row in table.items()},
    )
    (cell / "cv_rescore.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {cell / 'cv_rescore.json'} ({len(results)} models)")
    return out


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
