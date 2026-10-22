"""What fixing E's link temperature costs each theory (PI 2026-10-22).

E, the literature, is the reference paradigm: its link's temperature is fixed at 1
(`src.rsa_open.corpus.LITERATURE_FIXED`), because on E alone a free temperature traded off against
a theory's own sharpness and one fit stalled for hours. The worry is that a free temperature is
sometimes right: a theory that re-attributes its speaker's sharpness to a sharper listener choice
would be squashed. This audit fits every theory on E both ways and reports, per theory:

- the fits' time, divergences and convergence (a free fit that does not finish within the limit is
  reported as such: that is part of the finding);
- the posterior means, the temperature and the theory's own parameters side by side;
- the difference in expected log predictive density on E (PSIS-LOO, free minus fixed) with its SE
  clustered by display (`src.models.clustered_se.cluster_dse`): positive where the free
  temperature predicts E better.

Many theories clearly better with a free temperature would argue for a prior around 1 instead of
the fixed value; small differences, that the fix costs little.

    <venv python> -m src.rsa_open.temperature_audit --bar <the bar> --literature <all_trials.csv> \\
        --out <dir> [--time-limit-hours 4]
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import tyro

from src.models.model_manifest import read_manifest_entries
from src.rsa.fit import FitSettings
from src.rsa_open.corpus import Block, CorpusFit, literature_block

VARIANTS = ("fixed", "free")


def _loo_i(fitted: CorpusFit, block: str) -> Optional[np.ndarray]:
    import arviz as az

    ll = fitted.pointwise(block)  # (draws, trials)
    # arviz requires a posterior group; PSIS-LOO reads only the log-likelihood.
    idata = az.from_dict(posterior={"_": np.zeros((1, ll.shape[0]))}, log_likelihood={"y": ll[None]})
    loo = az.loo(idata, pointwise=True)
    return np.asarray(loo.loo_i)


def audit(theories: Mapping[str, Path], literature: Block, out: Path, *, settings: FitSettings,
          time_limit_sec: Optional[float], workers: int = 0) -> List[Dict[str, Any]]:
    """Every theory fitted on ``literature`` with its temperature fixed (as the campaign fits it)
    and free; one row per theory, also written to ``out/audit.json`` and ``out/audit.md``."""
    from src.models.clustered_se import cluster_dse
    from src.rsa.loop.fitting import ModelFailure
    from src.rsa.loop.orchestrator import default_fit_workers
    from src.rsa_open.loop.fitting import fit_cached, write_corpus

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    free = replace(literature, fixed=())
    fixed = literature if literature.fixed else replace(literature, fixed=(("temperature", 1.0),))
    corpora = {"fixed": write_corpus(out / "corpus_fixed.json", [fixed]),
               "free": write_corpus(out / "corpus_free.json", [free])}
    _, groups = np.unique(np.asarray(literature.units), return_inverse=True)

    def one(job):
        name, variant = job
        t0 = time.time()
        try:
            # Always in a child of its own (an infinite limit when none): numpyro's handlers are
            # process-global, so concurrent fits in one process corrupt each other's traces.
            f = fit_cached(Path(theories[name]), name, corpora[variant], settings, out / ".fit_cache",
                           time_limit_sec=math.inf if time_limit_sec is None else time_limit_sec)
        except ModelFailure as exc:
            return name, variant, None, f"{type(exc).__name__}: {exc}", time.time() - t0
        return name, variant, f, None, time.time() - t0

    jobs = [(n, v) for n in theories for v in VARIANTS]
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), workers or default_fit_workers()))) as pool:
        results = list(pool.map(one, jobs))
    by = {(n, v): (f, err, sec) for n, v, f, err, sec in results}
    rows = []
    for name in theories:
        row: Dict[str, Any] = {"theory": name}
        loo = {}
        for v in VARIANTS:
            f, err, sec = by[(name, v)]
            entry: Dict[str, Any] = {"seconds": round(sec), "error": err}
            if f is not None:
                post = f.idata.posterior
                entry.update(converged=f.converged, problems=f.convergence_problems,
                             divergences=int(np.asarray(f.idata.sample_stats["diverging"]).sum()),
                             means={k: round(float(post[k].mean()), 4) for k in post.data_vars},
                             sds={k: round(float(post[k].std()), 4) for k in post.data_vars})
                loo[v] = _loo_i(f, literature.name)
                entry["elpd_loo"] = round(float(loo[v].sum()), 2)
            row[v] = entry
        if len(loo) == 2:
            row["elpd_free_minus_fixed"] = round(float(loo["free"].sum() - loo["fixed"].sum()), 2)
            row["dse_clustered"] = round(cluster_dse(loo["free"], loo["fixed"], groups), 2)
        rows.append(row)
    (out / "audit.json").write_text(json.dumps(rows, indent=1))
    (out / "audit.md").write_text(report_md(rows, literature))
    return rows


def report_md(rows: List[Dict[str, Any]], literature: Block) -> str:
    lines = [f"# E's temperature: fixed at 1 or free ({literature.n} trials)", "",
             "| theory | fixed: s, div | free: s, div | free temperature | elpd free − fixed (± clustered SE) |",
             "|---|---|---|---|---|"]
    for r in rows:
        def cell(v):
            e = r[v]
            return f"failed: {e['error']}" if e.get("error") else f"{e['seconds']} s, {e['divergences']}" + \
                ("" if e.get("converged") else " (not converged)")
        t = r["free"].get("means", {}).get("temperature@E")
        diff = (f"{r['elpd_free_minus_fixed']:+.1f} ± {r['dse_clustered']:.1f}" if "elpd_free_minus_fixed" in r
                else "—")
        lines.append(f"| {r['theory']} | {cell('fixed')} | {cell('free')} | {'' if t is None else f'{t:.2f}'} | {diff} |")
    return "\n".join(lines) + "\n"


@dataclass
class Args:
    bar: Path
    literature: Path
    """E: every literature trial (promote's all_trials.csv)."""
    out: Path
    time_limit_hours: float = 4.0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    workers: int = 0


def main(args: Args) -> List[Dict[str, Any]]:
    theories = {e["name"]: Path(args.bar) / f"{e['name']}.py" for e in read_manifest_entries(args.bar)}
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           dense_mass=True)
    rows = audit(theories, literature_block(args.literature), args.out, settings=settings,
                 time_limit_sec=args.time_limit_hours * 3600, workers=args.workers)
    print((Path(args.out) / "audit.md").read_text())
    return rows


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
