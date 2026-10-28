"""Audit an ablation's fits for a second posterior mode the chains never saw.

The ablations (``src.ablation``, method ``nuts_no_jitter``) start all four chains at the model's
initial point, so R-hat cannot see a second mode: four chains in one wrong mode agree. The RSA
consolidation found exactly that (docs/auto_rsa/CONSOLIDATION.md §6): its full superset had two
modes, and for some variants a fixed start lay in the worse one.

For every model and fold of an ablation's work directory this samples nothing. It rebuilds the
model on the fold's training data as the worker did, optimises the log density (on the
unconstrained scale, as NUTS samples it) from the initial point the chains started at and from
``n_uniform`` uniform starts in (-2, 2), and groups the optima into modes (`src.rsa.fit.find_modes`:
one mode when a straight path between them rises at most 0.5 nats; Laplace masses with a
finite-difference Hessian). A fit is flagged when its start lay outside the best mode, or when
another mode holds more than 1% of the Laplace mass.

Only for models without person-level parameters: with hundreds of per-person effects the
highest-density point is not where the mass is, and a Laplace mass is not trustworthy. The
sr-oct26 finals all have them; the number-game final has none.

    python scripts/paper/ablation_mode_audit.py --out-dir $SCRATCH/auto-psych/ng_ablation/ng \\
        --folds-dir $SCRATCH/auto-psych/ng_ablation/data/.cv_folds/<digest> --report <json> --workers 8
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import tyro


@dataclass
class Args:
    out_dir: Path
    """The ablation's elimination directory (``models/``, ``cv/``)."""
    folds_dir: Path
    """The folds' training files (``train_fold<f>.csv``)."""
    report: Path
    models: Optional[List[str]] = None
    """Default: every model with a fold record under ``cv/``."""
    n_uniform: int = 20
    seed: int = 0
    workers: int = 1


def _finite_difference_hessian(grad, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    columns = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = step
        columns.append((grad(x + e) - grad(x - e)) / (2 * step))
    h = np.stack(columns, axis=1)
    return (h + h.T) / 2


def audit_one(out_dir: Path, folds_dir: Path, name: str, fold: int, n_uniform: int, seed: int) -> dict:
    import pymc as pm
    from pymc.blocking import DictToArrayBijection, RaveledVars

    from src.models.data_binding import extract_observed
    from src.models.model_loading import load_pymc_model
    from src.rsa.fit import _barrier_free, find_modes, minor_mode_problem, mode_shares

    model = load_pymc_model(name, out_dir / "models")
    with model:
        pm.set_data(extract_observed(folds_dir / f"train_fold{fold}.csv", model))
    value_vars = model.continuous_value_vars  # compile_dlogp's default, in its order
    if len(value_vars) != len(model.value_vars):
        raise ValueError(f"{name} has discrete parameters; the audit optimises continuous ones only")
    logp = model.compile_logp(jacobian=True)
    dlogp = model.compile_dlogp(jacobian=True)
    initial = model.initial_point()
    start = DictToArrayBijection.map({v.name: initial[v.name] for v in value_vars})

    def point(x):
        return DictToArrayBijection.rmap(RaveledVars(np.asarray(x, dtype=start.data.dtype), start.point_map_info))

    def energy_and_grad(x):
        p = point(x)
        e = -float(logp(p))
        return (e, -np.asarray(dlogp(p), dtype=np.float64)) if np.isfinite(e) else (np.inf, np.zeros(len(x)))

    def grad(x):
        return energy_and_grad(x)[1]

    rng = np.random.default_rng(seed)
    starts = [np.asarray(start.data, dtype=np.float64)] + list(rng.uniform(-2.0, 2.0, (n_uniform, len(start.data))))
    modes, failed = find_modes(energy_and_grad, lambda x: _finite_difference_hessian(grad, x), starts)
    from_start, _ = find_modes(energy_and_grad, lambda x: _finite_difference_hessian(grad, x), starts[:1])
    s = from_start[0]
    start_mode = next((i for i, m in enumerate(modes) if _barrier_free(energy_and_grad, m, s.x, s.energy)), None)
    problem = minor_mode_problem(modes, visited=[i == start_mode for i in range(len(modes))])
    record = json.loads((out_dir / "cv" / name / f"fold{fold}.json").read_text())
    return dict(model=name, fold=fold, fit_converged=record["converged"], n_params=len(start.data),
                start_mode=start_mode, start_below_best=s.energy - modes[0].energy, failed_starts=failed,
                modes=[dict(energy_above_best=m.energy - modes[0].energy, share=share, starts=m.starts)
                       for m, share in zip(modes, mode_shares(modes))],
                flagged=start_mode != 0 or problem is not None,
                problem=problem if start_mode == 0 else
                f"the chains' start lies in mode {start_mode}, {s.energy - modes[0].energy:.1f} nats below the best")


def _own_caches() -> None:
    """A worker's own arviz cache and PyTensor compile directory, set before either is imported:
    processes sharing them collide on arviz's daily marker and wait on one compile lock (CLAUDE.md)."""
    import os
    import tempfile

    root = tempfile.mkdtemp(prefix=f"mode_audit_{os.getpid()}_")
    os.environ["XDG_CACHE_HOME"] = os.path.join(root, "xdg")
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={os.path.join(root, 'pytensor')}"


def main(args: Args) -> list:
    names = args.models or sorted(p.name for p in (args.out_dir / "cv").iterdir() if p.is_dir())
    tasks = [(n, int(f.stem[4:])) for n in names for f in sorted((args.out_dir / "cv" / n).glob("fold*.json"))]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_own_caches) as pool:
        futures = [pool.submit(audit_one, args.out_dir, args.folds_dir, n, f, args.n_uniform, args.seed)
                   for n, f in tasks]
        results = []
        for fut in futures:
            r = fut.result()
            results.append(r)
            print(f"{r['model']} fold {r['fold']}: {len(r['modes'])} mode(s), start in mode {r['start_mode']}"
                  f"{'  FLAGGED: ' + r['problem'] if r['flagged'] else ''}", flush=True)
    args.report.write_text(json.dumps(results, indent=1))
    flagged = [r for r in results if r["flagged"]]
    print(f"{len(flagged)} of {len(results)} fits flagged; report {args.report}", flush=True)
    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
