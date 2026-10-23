"""What NUTS says about a theory's posterior geometry, in words a theorist can act on.

The rehearsal's too-slow and non-converging theories (run B, October 2026) were slow because of
their geometry, not their cost: on pragmods every theory costs 16-21 ms per leapfrog step, but
one whose speaker rationality and cost weight are 0.92 correlated needs twice the steps per
draw at half the step size. A theorist cannot see that from a timeout or an R-hat. So every fit
records its steps per draw and adapted step size (`src.rsa_open.corpus.fit_corpus`), and
`SamplerReport` names what they mean:

- **steps per draw**: the leapfrog steps NUTS took per draw (per iteration, the most any chain
  took: vectorized chains wait for the slowest), and the step size it adapted to. Many steps
  at a small step size mean a narrow, curved posterior;
- **correlated pairs**: parameters whose posterior draws have rank correlation |r| ≥
  `CORRELATION_FLAG`: the data constrain a combination of the two, not each one (a ridge);
- **projected fit time**: the self-check fits a fraction of the data briefly
  (`src.rsa_open.loop.check_candidate`); `projected_fit_seconds` takes its steps per draw to the
  loop's fit on all the data, at the cost of one step there (`step_seconds`: a leapfrog step is
  one gradient of the log density, timed for the loop's chains on all the data).

A fit killed at its time limit leaves its warmup's report (`warmup_report`) when warmup
finished, so a too-slow rejection can say why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

CORRELATION_FLAG = 0.85
MANY_STEPS_PER_DRAW = 10.0
"""On pragmods a well-identified theory takes about 7 steps per draw (rsa_l1_salience, step size
0.66); the bar theory with a 0.92 alpha-cost ridge takes 13 (step size 0.29)."""
GRADIENT_REPEATS = 20


def _columns(samples: Mapping[str, np.ndarray], names: Sequence[str], lead: int) -> Dict[str, np.ndarray]:
    """One column per scalar parameter (a vector parameter's entries as ``name[i]``), over the
    pooled draws; ``lead`` is the number of leading draw axes."""
    out = {}
    for name in names:
        if name not in samples:
            continue
        v = np.asarray(samples[name])
        n = int(np.prod(v.shape[:lead]))
        v = v.reshape(n, -1)
        for i in range(v.shape[1]):
            out[name if v.shape[1] == 1 else f"{name}[{i}]"] = v[:, i]
    return out


def correlated_pairs(samples: Mapping[str, np.ndarray], names: Sequence[str], *, lead: int = 2,
                     threshold: float = CORRELATION_FLAG) -> List[Tuple[str, str, float]]:
    """Pairs of parameters whose draws have Spearman |r| ≥ ``threshold``, strongest first. Rank
    correlation, so a parameter's scale (log or not) does not matter; a constant parameter
    (fixed, or a sampler stuck on one value) has none."""
    from scipy.stats import rankdata

    cols = _columns(samples, names, lead)
    keys = [k for k, v in cols.items() if np.ptp(v) > 0]
    if len(keys) < 2:
        return []
    ranks = np.stack([rankdata(cols[k]) for k in keys])
    r = np.corrcoef(ranks)
    out = [(keys[i], keys[j], float(r[i, j])) for i in range(len(keys)) for j in range(i + 1, len(keys))
           if abs(r[i, j]) >= threshold]
    return sorted(out, key=lambda t: -abs(t[2]))


@dataclass(frozen=True)
class SamplerReport:
    steps_per_draw: float
    step_size: float
    correlated: Tuple[Tuple[str, str, float], ...]
    phase: str = "sampling"
    """``sampling``: from the fit's draws; ``warmup``: from its warmup's second half (a fit killed
    while sampling)."""

    def lines(self) -> List[str]:
        where = "while sampling" if self.phase == "sampling" else "by the end of warmup"
        out = [f"NUTS took {self.steps_per_draw:.1f} leapfrog steps per draw {where}, at step size "
               f"{self.step_size:.3g}" + (
                   f": many (above about {MANY_STEPS_PER_DRAW:.0f}), so the posterior is narrow or curved, usually "
                   "because the data cannot separate some parameters" if self.steps_per_draw > MANY_STEPS_PER_DRAW
                   else "")]
        for a, b, r in self.correlated:
            out.append(f"{a} and {b} are correlated at {r:+.2f} across the posterior: the data pin down a combination "
                       f"of them, not each one. Drop or fix one, or reparameterise so that one parameter carries "
                       f"the combination, unless you can say what in the data separates them")
        return out

    def text(self) -> str:
        return "; ".join(self.lines())

    def to_json(self) -> Dict[str, Any]:
        return dict(steps_per_draw=self.steps_per_draw, step_size=self.step_size,
                    correlated=[list(c) for c in self.correlated], phase=self.phase)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SamplerReport":
        return cls(float(d["steps_per_draw"]), float(d["step_size"]),
                   tuple((str(a), str(b), float(r)) for a, b, r in d["correlated"]), str(d["phase"]))


def steps_per_draw(n_steps: np.ndarray) -> float:
    """(chains, draws) leapfrog steps → mean over draws of the most any chain took."""
    return float(np.asarray(n_steps).max(axis=0).mean())


def report_of(idata: Any, names: Sequence[str]) -> Optional[SamplerReport]:
    """A fit's report from its trace (None for a fit recorded before steps were kept)."""
    stats = idata.sample_stats
    if "n_steps" not in stats or "step_size" not in stats:
        return None
    post = {k: np.asarray(idata.posterior[k]) for k in idata.posterior.data_vars}
    return SamplerReport(steps_per_draw(np.asarray(stats["n_steps"])), float(np.median(np.asarray(stats["step_size"]))),
                         tuple(correlated_pairs(post, names)))


def warmup_report(samples: Mapping[str, np.ndarray], n_steps: np.ndarray, step_size: np.ndarray,
                  names: Sequence[str]) -> SamplerReport:
    """From a warmup collected by chain: steps per draw and correlations over its second half
    (the first explores from the initial point)."""
    half = np.asarray(n_steps).shape[1] // 2
    tail = {k: np.asarray(v)[:, half:] for k, v in samples.items()}
    return SamplerReport(steps_per_draw(np.asarray(n_steps)[:, half:]), float(np.median(np.asarray(step_size)[:, -1])),
                         tuple(correlated_pairs(tail, names)), "warmup")


def step_seconds(theory: Any, blocks: Sequence[Any], chains: int, repeats: int = GRADIENT_REPEATS) -> float:
    """Seconds per evaluation of the corpus's log density and gradient for ``chains`` vectorized
    chains (compiled first, then timed over ``repeats``): a leapfrog step's cost. On pragmods
    (rsa_l1_salience) it is 7.8 and 9.9 ms for 2 and 4 chains, against 7.7 and 9.3 ms per step
    measured while sampling."""
    import time

    import jax
    from numpyro.infer.util import initialize_model

    from src.rsa_open.corpus import _corpus_model

    info = initialize_model(jax.random.split(jax.random.PRNGKey(0), chains), _corpus_model,
                            model_args=(theory, list(blocks)))
    grad = jax.jit(jax.vmap(jax.value_and_grad(info.potential_fn)))
    z = info.param_info.z
    jax.block_until_ready(grad(z))
    t = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(grad(z))
    return (time.perf_counter() - t) / repeats


def projected_fit_seconds(timing: Mapping[str, float], check_settings: Any, full_settings: Any,
                          full_step_sec: float) -> float:
    """The loop's fit time on all the data: a short fit's leapfrog steps (``timing``,
    `CorpusFit.timing`) taken to the full fit's iterations, at ``full_step_sec`` per step
    (`step_seconds` on all the data, the full fit's chains). A warmup's first iterations explore
    from the start, at a cost of their own; the full fit's extra warmup iterations are counted at
    the sampling phase's steps per draw. Compilation (a minute or two) is not included.

    Checked on pragmods, the self-check fitting a fifth of its displays (two measurements each, at
    different machine loads): `rsa_l1_salience` projected 114-170 s against 168 s for the full
    fit; `isolated_graded_costly_l3` (its alpha-cost ridge) 846-1,005 s against 867 s. Scaling
    the short fit's own time instead overstated both 2-3x: the sampler's fixed cost per step
    weighs on a short fit of little data, hardly on the full one."""
    per_draw = timing["sampling_steps"] / check_settings.num_samples
    steps = (timing["warmup_steps"] + max(0, full_settings.num_warmup - check_settings.num_warmup) * per_draw
             + per_draw * full_settings.num_samples)
    return full_step_sec * steps
