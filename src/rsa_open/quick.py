"""A quick approximate fit for the theorists' self-check (PI 2026-10-28): the best-fitting values and
a Laplace approximation around them, in seconds, from several random starts.

    python -m src.rsa_open.loop.check_candidate <dir> ... --quick

Run C's Gemini 4 theorists spent their sessions on 16-48 exploratory fits of their own (scipy,
jax) and 19 of 23 ran out of time before writing a theory. This gives them what those scripts were
for: where the theory fits best, how sharply the data pin each parameter down, which parameters
trade off against each other, and whether different starts end in different places. It is for
exploring only: the loop scores every theory with NUTS, never with this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from src.rsa_open.corpus import Block, _corpus_model, check_blocks
from src.rsa_open.fit import TheoryModel

QUICK_STEPS = 1500
QUICK_RESTARTS = 4
QUICK_DRAWS = 1000
CORRELATED = 0.85
SAME_OPTIMUM_NATS = 2.0


@dataclass
class QuickFit:
    seconds: float
    best: Dict[str, float]
    sd: Dict[str, float]
    log_density: List[float]
    """The optimum each start reached (log joint density, best first)."""
    correlated: List[tuple] = field(default_factory=list)
    problem: Optional[str] = None

    def lines(self) -> List[str]:
        out = [f"Quick fit ({len(self.log_density)} starts, {self.seconds:.0f} s; an approximation for exploring, "
               "never how the loop scores):"]
        spread = self.log_density[0] - self.log_density[-1]
        if spread > SAME_OPTIMUM_NATS:
            out.append(f"- The starts ended {spread:.1f} nats apart (log density {', '.join(f'{v:.1f}' for v in self.log_density)}): "
                       "the posterior has more than one optimum, which NUTS samples slowly or not at all.")
        else:
            out.append("- Every start ended at the same optimum.")
        for k, v in self.best.items():
            sd = self.sd.get(k)
            out.append(f"- {k}: best {v:.3g}" + ("" if sd is None or not np.isfinite(sd) else f", ± {sd:.2g} (Laplace)"))
        if self.problem:
            out.append(f"- {self.problem}")
        for a, b, r in self.correlated:
            out.append(f"- {a} and {b} trade off (correlation {r:+.2f} at the optimum): the data pin down a "
                       "combination of them, not each.")
        return out


def _hessian(grad, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central differences of the gradient (2n gradient calls; no second compilation)."""
    cols = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h * max(1.0, abs(x[i]))
        cols.append((grad(x + e) - grad(x - e)) / (2 * e[i]))
    return np.stack(cols, axis=1)


def quick_fit(theory: TheoryModel, blocks: Sequence[Block], *, steps: int = QUICK_STEPS,
              restarts: int = QUICK_RESTARTS, seed: int = 0, lr: float = 0.05) -> QuickFit:
    """MAP from ``restarts`` random starts (Adam on the unconstrained space, every start at once, one
    compilation), then a Laplace approximation (the inverse Hessian) at the best one."""
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import init_to_uniform
    from numpyro.infer.util import initialize_model

    check_blocks(blocks)
    t0 = time.perf_counter()
    args = (theory, list(blocks))
    keys = jax.random.split(jax.random.PRNGKey(seed), restarts)
    info, potential_fn, postprocess_fn, _ = initialize_model(keys, _corpus_model, model_args=args,
                                                             init_strategy=init_to_uniform, dynamic_args=False)
    z0 = getattr(info, "param_info", info).z
    names = list(z0)
    _, unravel = ravel_pytree({k: v[0] for k, v in z0.items()})
    flat0 = jax.vmap(lambda z: ravel_pytree(z)[0])(z0)

    def loss(flat):
        return potential_fn(unravel(flat))

    value_and_grad = jax.jit(jax.value_and_grad(loss))  # one compilation, every start and the curvature

    def descend(x):
        m = v = np.zeros_like(x)
        for t in range(steps):
            _, g = value_and_grad(x)
            g = np.nan_to_num(np.asarray(g))
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            x = x - lr * (m / (1 - 0.9 ** (t + 1))) / (np.sqrt(v / (1 - 0.999 ** (t + 1))) + 1e-8)
        return x

    final = [descend(np.asarray(flat0[k])) for k in range(restarts)]
    energy = np.array([float(value_and_grad(x)[0]) for x in final])
    density = np.where(np.isfinite(energy), -energy, -np.inf)
    order = np.argsort(-density)
    best = final[order[0]]
    constrained = postprocess_fn(unravel(best))
    out = QuickFit(0.0, {k: float(np.asarray(constrained[k]).reshape(-1)[0]) if np.size(constrained[k]) == 1
                         else float("nan") for k in names}, {}, [float(density[i]) for i in order])
    for k in names:  # a vector parameter, entry by entry
        arr = np.asarray(constrained[k]).reshape(-1)
        if arr.size > 1:
            del out.best[k]
            out.best.update({f"{k}[{i}]": float(x) for i, x in enumerate(arr)})
    try:
        h = _hessian(lambda x: np.asarray(value_and_grad(x)[1]), best)
        evals = np.linalg.eigvalsh((h + h.T) / 2)
        if not np.all(np.isfinite(evals)) or evals.min() <= 1e-8 * max(1.0, abs(evals).max()):
            raise np.linalg.LinAlgError("not positive definite")
        chol = np.linalg.cholesky(np.linalg.inv((h + h.T) / 2))
        eps = np.random.default_rng(seed).standard_normal((QUICK_DRAWS, len(best)))
        draws = np.asarray(best)[None, :] + eps @ chol.T
        post = jax.vmap(lambda f: postprocess_fn(unravel(f)))(jnp.asarray(draws))
        cols, labels = [], []
        for k in names:
            arr = np.asarray(post[k]).reshape(QUICK_DRAWS, -1)
            for i in range(arr.shape[1]):
                cols.append(arr[:, i])
                labels.append(k if arr.shape[1] == 1 else f"{k}[{i}]")
        mat = np.stack(cols, axis=1)
        out.sd = {lab: float(mat[:, i].std()) for i, lab in enumerate(labels)}
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(mat, rowvar=False)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= CORRELATED:
                    out.correlated.append((labels[i], labels[j], float(corr[i, j])))
    except (np.linalg.LinAlgError, FloatingPointError) as exc:  # a finding, reported, not an error
        out.problem = (f"The curvature at the optimum is singular ({exc}): some direction in the parameters is "
                       "not pinned down by the data at all. Fix, drop or tie a parameter.")
    out.seconds = time.perf_counter() - t0
    return out
