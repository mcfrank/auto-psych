"""AIC and BIC of each domain's best models and seeds, beside the ELPD the paper uses.

Exploratory: the paper compares models by ELPD. This asks whether AIC or BIC
would have ordered the best models and the seeds differently.

- **Maximized log-likelihood**: each model rebuilt on the data it was fitted
  to and its observed log-likelihood alone (no priors, no Jacobians)
  maximized by L-BFGS over its unconstrained parameters, from its posterior
  mean and best posterior draw where a cached fit exists (and the model's
  initial point). A boundary MLE (a lapse at 0, a per-person sensitivity
  running off) is a supremum approached, reported with its gradient norm.
- **Parameter counts**, by perturbing each unconstrained coordinate and
  seeing which trials move: a coordinate that moves only one participant's
  trials is that participant's, one that moves several is population-level,
  one that moves none (an empty participant slot) is not counted. Two
  conventions: *conditional* counts every coordinate the likelihood depends
  on, per-person ones included; *marginal* integrates the per-person
  coordinates out and counts only the population-level ones. The rank of the
  Jacobian of the per-trial log-likelihood is reported beside the count: it
  is lower when parameters are redundant (``mu + sigma * z`` with ``z`` free).
- **Marginal likelihood** (models with per-person coordinates): each
  participant's coordinates integrated out under their prior by importance
  sampling from a Student-t around that participant's Laplace approximation,
  at the population values that maximize the Laplace-approximate marginal
  likelihood. An estimate of the marginal likelihood at one population value
  bounds its maximum from below.

BIC's ``n`` is the number of trials.

    uv run python -m scripts.paper.information_criteria --domain sr ...   (on Sherlock: the .sbatch)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats


def criteria(log_lik: float, k: int, n: int) -> Dict[str, float]:
    return {"aic": -2.0 * log_lik + 2.0 * k, "bic": -2.0 * log_lik + k * math.log(n)}


def unit_column(responses: Path, column: str) -> np.ndarray:
    with Path(responses).open(newline="") as f:
        return np.asarray([row[column] for row in csv.DictReader(f)], dtype=object)


# ---------------------------------------------------------------------------
# Likelihoods on the unconstrained scale
# ---------------------------------------------------------------------------

class PyMCLikelihood:
    """A PyMC model bound to its data; functions of the raveled unconstrained parameters."""

    def __init__(self, model: Any, n_rows: int) -> None:
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt

        self.model = model
        free = list(model.free_RVs)
        if any(rv.type.dtype.startswith(("int", "uint", "bool")) for rv in free):
            raise ValueError("a model with discrete parameters has no gradient to maximize")
        self.value_vars = [model.rvs_to_values[rv] for rv in free]
        point = model.initial_point()
        self.shapes = [np.shape(point[v.name]) for v in self.value_vars]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = sum(self.sizes)
        self.coordinate_names = [
            v.name if not shape else f"{v.name}[{','.join(map(str, np.unravel_index(i, shape)))}]"
            for v, shape, size in zip(self.value_vars, self.shapes, self.sizes) for i in range(size)]
        self.x0 = self.ravel(point)

        terms = model.logp(vars=model.observed_RVs, jacobian=False, sum=False)
        pointwise = pt.concatenate([pt.flatten(t) for t in terms])
        total = pointwise.sum()
        priors = model.logp(vars=free, jacobian=True, sum=False)
        # Elementwise prior of each coordinate (zeros for a multivariate prior,
        # whose density does not split by coordinate: it cannot be per-person).
        self.elementwise_prior = [p.ndim == len(s) for p, s in zip(priors, self.shapes)]
        prior_flat = pt.concatenate([pt.flatten(p) if ok else pt.zeros(size)
                                     for p, ok, size in zip(priors, self.elementwise_prior, self.sizes)])
        mask = pt.vector("mask")
        joint = total + (prior_flat * mask).sum()
        prior_total = sum(p.sum() for p in priors)
        posterior = total + prior_total

        def compile_(outs, extra=()):
            return pytensor.function([*self.value_vars, *extra], outs, on_unused_input="ignore",
                                     givens=None, mode=None)

        with model:
            self._pointwise = compile_(pointwise)
            self._total_grad = compile_([total, *pytensor.grad(total, self.value_vars)])
            self._joint_grad = compile_([joint, *pytensor.grad(joint, self.value_vars)], extra=[mask])
            self._parts = compile_([pointwise, prior_flat])
            self._posterior_grad = compile_([posterior, *pytensor.grad(posterior, self.value_vars)])
            self._prior_total = compile_(prior_total)
        self.n_trials = int(self.pointwise(self.x0).size)
        if self.n_trials != n_rows:
            raise ValueError(f"the model's observed log-likelihood has {self.n_trials} terms "
                             f"but the responses have {n_rows} rows")
        self._pm = pm

    @classmethod
    def build(cls, models_dir: Path, name: str, responses: Path) -> "PyMCLikelihood":
        import pymc as pm
        from src.models.data_binding import extract_observed
        from src.models.model_loading import load_pymc_model

        model = load_pymc_model(name, Path(models_dir))
        with model:
            pm.set_data(extract_observed(Path(responses), model))
        return cls(model, n_rows=len(unit_column_rows(responses)))

    def ravel(self, point: Mapping[str, Any]) -> np.ndarray:
        return np.concatenate([np.asarray(point[v.name], dtype=float).reshape(-1) for v in self.value_vars])

    def unravel(self, x: np.ndarray) -> List[np.ndarray]:
        out, start = [], 0
        for v, shape, size in zip(self.value_vars, self.shapes, self.sizes):
            out.append(np.asarray(x[start:start + size], dtype=v.type.dtype).reshape(shape))
            start += size
        return out

    def point_from_constrained(self, values: Mapping[str, Any]) -> np.ndarray:
        """Constrained parameter values (a posterior draw or mean) on the unconstrained scale."""
        from pymc.initial_point import make_initial_point_fn

        free = {rv.name for rv in self.model.free_RVs}
        missing = free - set(values)
        if missing:
            raise KeyError(f"no values for the free parameters {sorted(missing)}")
        fn = make_initial_point_fn(model=self.model, overrides={k: v for k, v in values.items() if k in free},
                                   jitter_rvs=set(), return_transformed=True)
        return self.ravel(fn(0))

    def pointwise(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._pointwise(*self.unravel(x)), dtype=float)

    def total_and_grad(self, x: np.ndarray):
        total, *grads = self._total_grad(*self.unravel(x))
        return float(total), np.concatenate([np.asarray(g, dtype=float).reshape(-1) for g in grads])

    def joint_and_grad(self, x: np.ndarray, mask: np.ndarray):
        joint, *grads = self._joint_grad(*self.unravel(x), mask)
        return float(joint), np.concatenate([np.asarray(g, dtype=float).reshape(-1) for g in grads])

    def parts(self, x: np.ndarray):
        pointwise, prior = self._parts(*self.unravel(x))
        return np.asarray(pointwise, dtype=float), np.asarray(prior, dtype=float)

    def posterior_and_grad(self, x: np.ndarray):
        """The log-likelihood plus the whole log prior (with the transforms' Jacobians)."""
        value, *grads = self._posterior_grad(*self.unravel(x))
        return float(value), np.concatenate([np.asarray(g, dtype=float).reshape(-1) for g in grads])

    def log_prior(self, x: np.ndarray) -> float:
        return float(self._prior_total(*self.unravel(x)))

    def has_elementwise_prior(self, coordinate: int) -> bool:
        start = 0
        for ok, size in zip(self.elementwise_prior, self.sizes):
            if coordinate < start + size:
                return ok
            start += size
        raise IndexError(coordinate)


def unit_column_rows(responses: Path) -> List[Dict[str, str]]:
    with Path(responses).open(newline="") as f:
        return list(csv.DictReader(f))


class RSALikelihood:
    """A memo RSA model's log-likelihood of forced choices (``src.rsa.fit``'s), on the
    unconstrained scale of its scalar parameters."""

    def __init__(self, model: Any, prep: Any) -> None:
        import jax
        import jax.numpy as jnp
        from numpyro.distributions import biject_to

        from src.rsa.fit import _padded_probs, _safe_logits

        self.names = list(model.params)
        dists = [model.params[n] for n in self.names]
        if any(d.batch_shape != () or d.event_shape != () for d in dists):
            raise ValueError(f"{model.name}: a non-scalar parameter")
        transforms = [biject_to(d.support) for d in dists]
        self.dim = len(self.names)
        self.coordinate_names = list(self.names)
        self.n_trials = int(prep.n_trials)
        rows, classes = jnp.asarray(prep.pattern_row), jnp.asarray(prep.pattern_class)
        counts = jnp.asarray(prep.pattern_count, dtype=jnp.result_type(float))

        def params_of(x):
            return {n: t(x[i]) for i, (n, t) in enumerate(zip(self.names, transforms))}

        def probs(x):
            return _padded_probs(model, params_of(x), prep.groups, prep.width)

        def total(x):
            return jnp.sum(counts * _safe_logits(probs(x))[rows, classes])

        def log_prior(x):
            return sum(d.log_prob(t(x[i])) + t.log_abs_det_jacobian(x[i], t(x[i]))
                       for i, (d, t) in enumerate(zip(dists, transforms)))

        self._value_and_grad = jax.jit(jax.value_and_grad(total))
        self._posterior_value_and_grad = jax.jit(jax.value_and_grad(lambda x: total(x) + log_prior(x)))
        self._probs = jax.jit(lambda x: probs(x).reshape(-1))
        keys = jax.random.split(jax.random.PRNGKey(0), self.dim)
        self.x0 = np.asarray([float(t.inv(jnp.median(d.sample(k, (101,)))))
                              for d, t, k in zip(dists, transforms, keys)])
        self._transforms = transforms

    @classmethod
    def build(cls, model: Any, contexts: Sequence[Any], choices: Sequence[int]) -> "RSALikelihood":
        from src.rsa.fit import prepare
        from src.rsa.model_file import check_contract

        prep = prepare(contexts, choices)
        check_contract(model, {g.shape: g for g in prep.groups})
        return cls(model, prep)

    def point_from_constrained(self, values: Mapping[str, Any]) -> np.ndarray:
        return np.asarray([float(t.inv(values[n])) for n, t in zip(self.names, self._transforms)])

    def total_and_grad(self, x: np.ndarray):
        import jax.numpy as jnp

        value, grad = self._value_and_grad(jnp.asarray(x, dtype=jnp.result_type(float)))
        return float(value), np.asarray(grad, dtype=float)

    def posterior_and_grad(self, x: np.ndarray):
        import jax.numpy as jnp

        value, grad = self._posterior_value_and_grad(jnp.asarray(x, dtype=jnp.result_type(float)))
        return float(value), np.asarray(grad, dtype=float)

    def class_probs(self, x: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        return np.asarray(self._probs(jnp.asarray(x, dtype=jnp.result_type(float))), dtype=float)


# ---------------------------------------------------------------------------
# Maximum likelihood, parameter counts
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    log_lik: float
    x: np.ndarray
    grad_norm: float
    message: str
    start: int  # which start reached it
    iterations: int = 0
    trajectory: List[tuple] = field(default_factory=list)  # (iteration, log-likelihood)


TRAJECTORY_EVERY = 10_000


def _maximize_from(f_and_grad: Callable, x0: np.ndarray, free: np.ndarray, maxiter: int = 100_000,
                   log: Optional[Callable[[str], None]] = None):
    def neg(v):
        x = x0.copy()
        x[free] = v
        value, grad = f_and_grad(x)
        if not math.isfinite(value):
            return 1e300, np.zeros(len(free))
        return -value, -grad[free]

    trajectory, count = [], [0]

    def callback(intermediate_result):  # scipy passes the OptimizeResult only under this name
        count[0] += 1
        if count[0] % TRAJECTORY_EVERY == 0:
            trajectory.append((count[0], -float(intermediate_result.fun)))
            if log is not None:
                log(f"iteration {count[0]}: log-likelihood {trajectory[-1][1]:.4f}")

    res = optimize.minimize(neg, x0[free], jac=True, method="L-BFGS-B", callback=callback,
                            options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-14, "gtol": 1e-7})
    x = x0.copy()
    x[free] = res.x
    trajectory.append((int(res.nit), -float(res.fun)))
    return -float(res.fun), x, float(np.linalg.norm(res.jac)), str(res.message), int(res.nit), trajectory


def maximize(lik: Any, starts: Sequence[np.ndarray], free: Optional[np.ndarray] = None,
             maxiter: int = 100_000, log: Optional[Callable[[str], None]] = None) -> Fit:
    """The highest log-likelihood L-BFGS reaches from ``starts`` over the coordinates ``free``."""
    free = np.arange(lik.dim) if free is None else np.asarray(free)
    best = None
    for i, x0 in enumerate(starts):
        value, x, gnorm, message, nit, trajectory = _maximize_from(
            lik.total_and_grad, np.asarray(x0, dtype=float), free, maxiter=maxiter, log=log)
        if log is not None:
            log(f"start {i}: log-likelihood {value:.4f} after {nit} iterations ({message})")
        if best is None or value > best.log_lik:
            best = Fit(log_lik=value, x=x, grad_norm=gnorm, message=message, start=i,
                       iterations=nit, trajectory=trajectory)
    if best is None:
        raise ValueError("no starting points")
    return best


@dataclass
class Coordinates:
    population: List[int]
    per_unit: Dict[Any, List[int]]
    unused: List[int]

    @property
    def active(self) -> np.ndarray:
        return np.asarray(sorted(self.population + [i for idx in self.per_unit.values() for i in idx]), dtype=int)

    @property
    def k_conditional(self) -> int:
        return len(self.active)

    @property
    def k_marginal(self) -> int:
        return len(self.population)


def generic(x: np.ndarray, seed: int = 0, sd: float = 0.3) -> np.ndarray:
    """``x`` jittered, so no coordinate sits where another's effect vanishes
    (at ``z = 0`` a non-centered ``sigma`` moves nothing)."""
    return np.asarray(x, dtype=float) + np.random.default_rng(seed).normal(0.0, sd, np.shape(x))


def classify_coordinates(lik: PyMCLikelihood, x: np.ndarray, units: Optional[np.ndarray],
                         delta: float = 0.37) -> Coordinates:
    x = generic(x)
    base = lik.pointwise(x)
    population, unused, per_unit = [], [], {}
    for j in range(lik.dim):
        moved = x.copy()
        moved[j] += delta
        changed = np.abs(lik.pointwise(moved) - base) > 1e-10 * (1.0 + np.abs(base))
        if not changed.any():
            unused.append(j)
            continue
        owners = set(units[changed]) if units is not None else None
        if owners is not None and len(owners) == 1:
            per_unit.setdefault(owners.pop(), []).append(j)
        else:
            population.append(j)
    return Coordinates(population=population, per_unit=per_unit, unused=unused)


def jacobian_rank(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, idx: Sequence[int],
                  h: float = 1e-4, rtol: float = 1e-6) -> int:
    """Numerical rank of d fn / d x[idx] (central differences)."""
    columns = []
    for j in idx:
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        columns.append((fn(up) - fn(down)) / (2 * h))
    s = np.linalg.svd(np.stack(columns, axis=1), compute_uv=False)
    return int(np.sum(s > s[0] * rtol)) if s.size and s[0] > 0 else 0


# ---------------------------------------------------------------------------
# Marginal likelihood over per-person coordinates
# ---------------------------------------------------------------------------

@dataclass
class Marginal:
    log_lik: float  # importance-sampling estimate at the population values below
    laplace: float  # the Laplace approximation there
    min_ess: float  # the smallest per-person effective sample size
    x: np.ndarray  # the population values (and each person's Laplace mode)
    n_samples: int


@dataclass
class _Blocks:
    units: List[Any]
    coords: np.ndarray  # (persons, d)
    trial_unit: np.ndarray  # (trials,) the person's row in ``units``, -1 for none
    mask: np.ndarray  # (dim,) 1 at per-person coordinates
    warm: Optional[np.ndarray] = None


def _blocks(lik: PyMCLikelihood, coords: Coordinates, units: np.ndarray) -> _Blocks:
    names = list(coords.per_unit)
    sizes = {len(v) for v in coords.per_unit.values()}
    if len(sizes) != 1:
        raise ValueError(f"participants have differing numbers of own coordinates: {sorted(sizes)}")
    matrix = np.asarray([coords.per_unit[u] for u in names], dtype=int)
    bad = [lik.coordinate_names[j] for j in matrix.reshape(-1) if not lik.has_elementwise_prior(j)]
    if bad:
        raise ValueError(f"per-person coordinates without an elementwise prior: {bad[:5]}")
    row = {u: i for i, u in enumerate(names)}
    mask = np.zeros(lik.dim)
    mask[matrix.reshape(-1)] = 1.0
    return _Blocks(units=names, coords=matrix, trial_unit=np.asarray([row.get(u, -1) for u in units]), mask=mask)


def _person_terms(lik: PyMCLikelihood, b: _Blocks, x: np.ndarray):
    """Each person's log-likelihood plus the prior of their coordinates, and the
    log-likelihood of trials no person owns."""
    pointwise, prior = lik.parts(x)
    own = b.trial_unit >= 0
    per = np.bincount(b.trial_unit[own], weights=pointwise[own], minlength=len(b.units))
    per += prior[b.coords].sum(axis=1)
    return per, float(pointwise[~own].sum())


def _laplace_at(lik: PyMCLikelihood, b: _Blocks, x: np.ndarray, h: float = 1e-4):
    """Each person's Laplace mode and the negative Hessian there, with the
    population coordinates as in ``x``."""
    flat = b.coords.reshape(-1)
    start = x.copy()
    if b.warm is not None:
        start[flat] = b.warm
    _, mode, *_ = _maximize_from(lambda y: lik.joint_and_grad(y, b.mask), start, flat)
    b.warm = mode[flat].copy()
    d = b.coords.shape[1]
    neg_h = np.empty((len(b.units), d, d))
    for r in range(d):
        up, down = mode.copy(), mode.copy()
        up[b.coords[:, r]] += h
        down[b.coords[:, r]] -= h
        diff = (lik.joint_and_grad(up, b.mask)[1] - lik.joint_and_grad(down, b.mask)[1]) / (2 * h)
        neg_h[:, :, r] = -diff[b.coords]
    neg_h = (neg_h + np.transpose(neg_h, (0, 2, 1))) / 2
    return mode, neg_h


def _laplace_total(lik: PyMCLikelihood, b: _Blocks, mode: np.ndarray, neg_h: np.ndarray) -> float:
    per, rest = _person_terms(lik, b, mode)
    sign, logdet = np.linalg.slogdet(neg_h)
    if np.any(sign <= 0):
        bad = [b.units[i] for i in np.where(sign <= 0)[0][:5]]
        raise RuntimeError(f"the Hessian at the Laplace mode is not negative definite for {bad}")
    d = neg_h.shape[1]
    return rest + float(np.sum(per + 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet))


def marginal_log_lik(lik: PyMCLikelihood, coords: Coordinates, units: np.ndarray, *, start: np.ndarray,
                     n_samples: int, seed: int, df: float = 4.0, scale: float = 1.3) -> Marginal:
    b = _blocks(lik, coords, units)
    pop = np.asarray(coords.population, dtype=int)
    current = np.asarray(start, dtype=float).copy()

    def neg_laplace(v):
        x = current.copy()
        x[pop] = v
        mode, neg_h = _laplace_at(lik, b, x)
        return -_laplace_total(lik, b, mode, neg_h)

    res = optimize.minimize(neg_laplace, current[pop], method="L-BFGS-B",
                            options={"maxiter": 2000, "eps": 1e-5, "ftol": 1e-12, "gtol": 1e-6})
    current[pop] = res.x
    mode, neg_h = _laplace_at(lik, b, current)
    laplace = _laplace_total(lik, b, mode, neg_h)

    total, min_ess = _person_is(lik, b, mode, neg_h, n_samples, np.random.default_rng(seed), df, scale)
    return Marginal(log_lik=total, laplace=laplace, min_ess=min_ess, x=mode, n_samples=n_samples)


def _mvt_log_density(maha: np.ndarray, logdet: np.ndarray, d: int, df: float) -> np.ndarray:
    return (special.gammaln((df + d) / 2) - special.gammaln(df / 2) - 0.5 * d * math.log(df * math.pi)
            - 0.5 * logdet - 0.5 * (df + d) * np.log1p(maha / df))


def _person_is(lik: PyMCLikelihood, b: _Blocks, mode: np.ndarray, neg_h: np.ndarray, n_samples: int,
               rng: np.random.Generator, df: float, scale: float):
    """Importance-sampled log of the likelihood with every person's coordinates
    integrated against their prior (a multivariate t around each person's
    Laplace mode), the population coordinates as in ``mode``; and the smallest
    person's effective sample size."""
    persons, d = b.coords.shape
    cov = scale ** 2 * np.linalg.inv(neg_h)
    chol = np.linalg.cholesky(cov)
    centre = mode[b.coords]  # (persons, d)
    logdet = 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    log_w = np.empty((n_samples, persons))
    rest = 0.0
    for s in range(n_samples):
        g = rng.standard_normal((persons, d))
        w = rng.chisquare(df, persons) / df
        u = centre + np.einsum("pij,pj->pi", chol, g) / np.sqrt(w)[:, None]
        x = mode.copy()
        x[b.coords] = u
        per, rest = _person_terms(lik, b, x)
        log_w[s] = per - _mvt_log_density(np.einsum("pi,pi->p", g, g) / w, logdet, d, df)
    log_person = special.logsumexp(log_w, axis=0) - math.log(n_samples)
    weights = np.exp(log_w - log_w.max(axis=0))
    ess = weights.sum(axis=0) ** 2 / (weights ** 2).sum(axis=0)
    return rest + float(log_person.sum()), float(ess.min())


# ---------------------------------------------------------------------------
# Marginal likelihood under the priors (Bayes factors)
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    log_evidence: float  # importance-sampled log p(data | model)
    laplace: float
    min_ess: float  # of the (outer) importance weights
    n_samples: int
    inner_min_ess: Optional[float] = None  # hierarchical: the worst person's, over the outer samples


def _hessian_from_grad(grad: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    out = np.empty((len(x), len(x)))
    for j in range(len(x)):
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        out[:, j] = (grad(up) - grad(down)) / (2 * h)
    return (out + out.T) / 2


def _hessian_from_values(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 5e-3) -> np.ndarray:
    d = len(x)
    out = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            def at(si, sj):
                y = x.copy()
                y[i] += si * h
                y[j] += sj * h
                return f(y)
            out[i, j] = out[j, i] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (4 * h * h)
    return out


def _laplace_log_integral(value: float, neg_h: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(neg_h)
    if sign <= 0:
        raise RuntimeError("the Hessian at the posterior mode is not negative definite")
    return value + 0.5 * len(neg_h) * math.log(2 * math.pi) - 0.5 * logdet


def _t_importance(log_f: Callable[[np.ndarray], float], mode: np.ndarray, neg_h: np.ndarray, n: int,
                  rng: np.random.Generator, df: float, scale: float):
    """log of the integral of exp(log_f) by importance sampling from a
    multivariate t around ``mode``, and the effective sample size."""
    d = len(mode)
    chol = np.linalg.cholesky(scale ** 2 * np.linalg.inv(neg_h))
    logdet = 2 * float(np.log(np.diag(chol)).sum())
    log_w = np.empty(n)
    for s in range(n):
        g = rng.standard_normal(d)
        w = rng.chisquare(df) / df
        value = log_f(mode + chol @ g / math.sqrt(w))
        log_w[s] = (value if math.isfinite(value) else -np.inf) - float(
            _mvt_log_density(np.asarray(g @ g / w), np.asarray(logdet), d, df))
    weights = np.exp(log_w - log_w.max())
    return float(special.logsumexp(log_w) - math.log(n)), float(weights.sum() ** 2 / (weights ** 2).sum())


def flat_log_evidence(lik: Any, *, start: np.ndarray, n_samples: int, seed: int,
                      df: float = 4.0, scale: float = 1.3) -> Evidence:
    """log p(data | model) of a model without per-person coordinates: the
    likelihood integrated against the whole prior over every coordinate, by
    importance sampling around the posterior mode."""
    _, mode, *_ = _maximize_from(lik.posterior_and_grad, np.asarray(start, dtype=float), np.arange(lik.dim))
    neg_h = -_hessian_from_grad(lambda x: lik.posterior_and_grad(x)[1], mode)
    laplace = _laplace_log_integral(lik.posterior_and_grad(mode)[0], neg_h)
    value, ess = _t_importance(lambda x: lik.posterior_and_grad(x)[0], mode, neg_h, n_samples,
                               np.random.default_rng(seed), df, scale)
    return Evidence(log_evidence=value, laplace=laplace, min_ess=ess, n_samples=n_samples)


def hierarchical_log_evidence(lik: PyMCLikelihood, coords: Coordinates, units: np.ndarray, *,
                              start: np.ndarray, n_outer: int, n_inner: int, seed: int,
                              df: float = 4.0, scale: float = 1.3) -> Evidence:
    """log p(data | model) with per-person coordinates: each person's
    coordinates integrated against their prior given the population ones
    (``_person_is``), and the population coordinates against theirs, by
    importance sampling around the mode of that marginal (nested: each outer
    draw's likelihood is the product of the persons' inner estimates, so the
    estimate is unbiased on the natural scale). Coordinates the likelihood does
    not use integrate to one under their own (elementwise) prior."""
    b = _blocks(lik, coords, units)
    bad = [lik.coordinate_names[j] for j in coords.unused if not lik.has_elementwise_prior(j)]
    if bad:
        raise ValueError(f"unused coordinates without an elementwise prior: {bad[:5]}")
    own = np.concatenate([b.coords.reshape(-1), np.asarray(coords.unused, dtype=int)])
    pop = np.asarray(coords.population, dtype=int)
    base = np.asarray(start, dtype=float).copy()

    def population_prior(x):
        return lik.log_prior(x) - float(lik.parts(x)[1][own].sum())

    def at(v):
        x = base.copy()
        x[pop] = v
        return x

    def laplace_marginal(v):
        mode, neg_h = _laplace_at(lik, b, at(v))
        return population_prior(mode) + _laplace_total(lik, b, mode, neg_h)

    res = optimize.minimize(lambda v: -laplace_marginal(v), base[pop], method="L-BFGS-B",
                            options={"maxiter": 2000, "eps": 1e-5, "ftol": 1e-12, "gtol": 1e-6})
    centre = res.x
    neg_h = -_hessian_from_values(laplace_marginal, centre)
    laplace = _laplace_log_integral(laplace_marginal(centre), neg_h)
    rng = np.random.default_rng(seed)
    inner_ess = []

    def outer(v):
        mode, person_h = _laplace_at(lik, b, at(v))
        total, ess = _person_is(lik, b, mode, person_h, n_inner, rng, df, scale)
        inner_ess.append(ess)
        return population_prior(mode) + total

    value, ess = _t_importance(outer, centre, neg_h, n_outer, rng, df, scale)
    return Evidence(log_evidence=value, laplace=laplace, min_ess=ess, n_samples=n_outer,
                    inner_min_ess=float(min(inner_ess)))


# ---------------------------------------------------------------------------
# One model's record
# ---------------------------------------------------------------------------

def _running_off(lik: Any, x: np.ndarray, beyond: float = 12.0) -> List[str]:
    """Coordinates the MLE pushed far out on the unconstrained scale (a supremum being approached)."""
    return [lik.coordinate_names[j] for j in np.where(np.abs(x) > beyond)[0]]


def pymc_record(models_dir: Path, name: str, responses: Path, *, unit: Optional[str],
                starts: Sequence[np.ndarray] = (), n_samples: int = 4000, seed: int = 0,
                constrained_starts: Sequence[Mapping[str, Any]] = (), jitter_starts: int = 0,
                maxiter: int = 2_000_000, evidence_outer: int = 1000,
                evidence_inner: int = 1000) -> Dict[str, Any]:
    lik = PyMCLikelihood.build(models_dir, name, responses)
    starts = [*starts, *(lik.point_from_constrained(v) for v in constrained_starts),
              *(generic(lik.x0, seed=s, sd=0.5) for s in range(1, jitter_starts + 1))]
    reference = starts[0] if starts else lik.x0
    units = unit_column(responses, unit) if unit else None
    coords = classify_coordinates(lik, reference, units)
    # A posterior's mean and best draw start nearer the maximum than the
    # model's initial point, which then only costs time.
    fit = maximize(lik, starts if constrained_starts else [*starts, lik.x0], maxiter=maxiter,
                   log=lambda m: print(f"[ic] {name}: {m}", flush=True))
    conditional = {
        "log_lik": fit.log_lik, "k": coords.k_conditional,
        "jacobian_rank": jacobian_rank(lik.pointwise, generic(reference), coords.active),
        "grad_norm": fit.grad_norm, "optimizer": fit.message, "iterations": fit.iterations,
        "trajectory": fit.trajectory, "running_off": _running_off(lik, fit.x),
        **criteria(fit.log_lik, coords.k_conditional, lik.n_trials),
    }
    marginal = None
    if coords.per_unit:
        m = marginal_log_lik(lik, coords, units, start=reference, n_samples=n_samples, seed=seed)
        marginal = {"log_lik": m.log_lik, "laplace": m.laplace, "min_ess": m.min_ess,
                    "n_samples": m.n_samples, "k": coords.k_marginal,
                    "is_lower_bound": True, **criteria(m.log_lik, coords.k_marginal, lik.n_trials)}
        ev = hierarchical_log_evidence(lik, coords, units, start=m.x, n_outer=evidence_outer,
                                       n_inner=evidence_inner, seed=seed)
    else:
        ev = flat_log_evidence(lik, start=fit.x, n_samples=n_samples, seed=seed)
    return {"model": name, "n": lik.n_trials, "n_units": len(coords.per_unit),
            "per_unit_k": (len(next(iter(coords.per_unit.values()))) if coords.per_unit else 0),
            "unused_coordinates": len(coords.unused), "conditional": conditional, "marginal": marginal,
            "evidence": _evidence_record(ev, evidence_inner if coords.per_unit else None)}


def _evidence_record(ev: Evidence, inner_samples: Optional[int]) -> Dict[str, Any]:
    return {"log_evidence": ev.log_evidence, "laplace": ev.laplace, "min_ess": ev.min_ess,
            "n_samples": ev.n_samples, "inner_min_ess": ev.inner_min_ess, "inner_samples": inner_samples}


# ---------------------------------------------------------------------------
# The paper's models: tasks, records, comparisons
# ---------------------------------------------------------------------------

from pyprojroot import here  # noqa: E402

FIGURE_DATA = here() / "data" / "paper" / "figure_data"
NG_UNSEEN = here() / "data" / "paper" / "number_game_unseen_items.json"
RSA_LIVE = here() / "data" / "rsa" / "live"
RSA_SEEDS = here() / "data" / "rsa" / "live_seeds"
DOMAIN_UNIT = {"subjective_randomness": "participant_id", "number_game": None}


@dataclass(frozen=True)
class Task:
    domain: str
    run: str  # "pooled" for the RSA campaign's pooled data
    model: str
    role: Literal["winner", "seed"]
    file: Path
    responses: Optional[Path] = None
    sources: Tuple[Path, ...] = ()  # pooled: each run's final responses (``pooled_responses``)


def _runs(root: Path) -> List[str]:
    return sorted(p.name for p in root.glob("run*") if p.is_dir())


def _pymc_tasks(domain: str) -> List[Task]:
    """Every run's winner and the seeds, on all runs' final data pooled (as
    Figures 2 and 3 fit them)."""
    from scripts.paper import number_game_figure_data as ng
    from scripts.paper import sr_figure_data as sr

    root = sr.DEFAULT_RUNS_ROOT if domain == "subjective_randomness" else ng.DEFAULT_RUNS_ROOT
    winners, seeds, sources = [], {}, []
    for run in _runs(root):
        tree = root / run / domain
        final = sr.final_experiment_dir(tree)
        loop = final / "model_loop"
        winner = json.loads((loop / "model_posterior.json").read_text())["best_model"]
        winners.append((winner, loop / "models" / f"{winner}.py"))
        for name in json.loads((tree / "starting_models.json").read_text())["starting_models"]:
            path = sr.seed_file(tree, final, name)
            if name in seeds and _sha(seeds[name]) != _sha(path):
                raise RuntimeError(f"seed {name} differs between runs: {seeds[name]} and {path}")
            seeds.setdefault(name, path)
        sources.append(loop / "responses.csv")
    if len({w for w, _ in winners}) != len(winners):
        raise RuntimeError(f"two runs share a winner's name: {winners}")
    return ([Task(domain, "pooled", w, "winner", f, sources=tuple(sources)) for w, f in winners]
            + [Task(domain, "pooled", n, "seed", f, sources=tuple(sources)) for n, f in seeds.items()])


def pooled_responses(task: Task, work_dir: Path) -> Path:
    """Each run's final responses in one CSV under ``work_dir``, pooled as the
    figure data pool them (subjective randomness renumbers each run's people)."""
    from scripts.paper import number_game_figure_data as ng
    from scripts.paper import sr_figure_data as sr

    out = Path(work_dir) / "pooled" / task.domain / "responses.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = sr.pool_responses if task.domain == "subjective_randomness" else ng.pool_responses
    pool({str(p): p for p in task.sources}, out)
    return out


def _rsa_tasks() -> List[Task]:
    out = []
    for chain in sorted(RSA_LIVE.glob("live_rsa_c*")):
        experiments = sorted((chain / "outputs").glob("experiment*"), key=lambda p: int(p.name[10:]))
        loop = experiments[-1] / "model_loop"
        best = json.loads((loop / "export.json").read_text())["best_model"]
        out.append(Task("rsa", "pooled", best, "winner", loop / "models" / f"{best}.py"))
    chains = json.loads((RSA_SEEDS / "chains" / "chains.json").read_text())["chains"]
    seeds = sorted({m for c in chains for m in c["models"]})
    out += [Task("rsa", "pooled", s, "seed", RSA_SEEDS / "models" / f"{s}.py") for s in seeds]
    return out


def tasks() -> List[Task]:
    return [*_pymc_tasks("subjective_randomness"), *_pymc_tasks("number_game"), *_rsa_tasks()]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cached_fit(fit_dir: Path, run: str, model: str) -> Path:
    found = sorted((fit_dir / run / "fit_cache").glob(f"{model}.*.nc"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one cached fit of {model} in {fit_dir / run / 'fit_cache'}, found {found}")
    return found[0]


def _posterior_summary(nc: Path) -> Dict[str, Any]:
    """PSIS-LOO and WAIC of a cached fit, and its posterior mean and best draw as starts."""
    import arviz as az

    idata = az.from_netcdf(str(nc))
    loo, waic = az.loo(idata), az.waic(idata)
    ll = idata.log_likelihood["response"]
    per_draw = ll.sum(dim=[d for d in ll.dims if d not in ("chain", "draw")])
    c, d = np.unravel_index(int(np.argmax(per_draw.values)), per_draw.shape)
    post = idata.posterior
    mean = {v: post[v].mean(("chain", "draw")).values for v in post.data_vars}
    best = {v: post[v].isel(chain=c, draw=d).values for v in post.data_vars}
    stats_ = {"elpd_loo": float(loo.elpd_loo), "p_loo": float(loo.p_loo),
              "lppd": float(loo.elpd_loo + loo.p_loo), "elpd_waic": float(waic.elpd_waic),
              "p_waic": float(waic.p_waic), "best_draw_log_lik": float(per_draw.values.max())}
    return {"stats": stats_, "starts": [mean, best]}


def run_task(task: Task, *, work_dir: Path, sr_fit_dir: Path, rsa_pooled: Optional[Path],
             n_samples: int) -> Dict[str, Any]:
    if task.domain == "rsa":
        record = rsa_record(task, rsa_pooled, n_samples=n_samples)
    else:
        responses = task.responses or pooled_responses(task, work_dir)
        posterior = None
        if task.domain == "subjective_randomness":
            fitted_on = sr_fit_dir / task.run / "responses.csv"
            if _sha(fitted_on) != _sha(responses):
                raise RuntimeError(f"{responses} is not the data the cached fits were fitted on ({fitted_on})")
            posterior = _posterior_summary(_cached_fit(sr_fit_dir, task.run, task.model))
        record = pymc_record(task.file.parent, task.model, responses, unit=DOMAIN_UNIT[task.domain],
                             constrained_starts=posterior["starts"] if posterior else (),
                             jitter_starts=0 if posterior else 4, n_samples=n_samples)
        if posterior:
            record["posterior"] = posterior["stats"]
        record["inputs"] = {"responses_sha256": _sha(responses)}
    record.update(domain=task.domain, run=task.run, role=task.role)
    record.setdefault("inputs", {}).update(model_sha256=_sha(task.file), script_sha256=_sha(Path(__file__)))
    return record


def rsa_record(task: Task, pooled: Optional[Path], *, n_samples: int) -> Dict[str, Any]:
    from src.rsa.dataset import load_forced_choice
    from src.rsa.model_file import RSAModel

    if pooled is None:
        raise ValueError("the RSA models need --rsa-pooled (the campaign's pooled data, on Sherlock only)")
    import jax

    # float64: in float32 the summed log-likelihood's rounding (~0.005 nats at
    # 70,000 trials) stops L-BFGS short of the maximum.
    jax.config.update("jax_enable_x64", True)
    bar = json.loads((RSA_LIVE / "bar_pool_preds.json").read_text())
    if _sha(pooled) != bar["data"]["sha256"]:
        raise RuntimeError(f"{pooled} is not the pooled data the bar was fitted on")
    trials = load_forced_choice(pooled)
    lik = RSALikelihood.build(RSAModel(task.file, name=task.model), trials.contexts, trials.choices)
    rng = np.random.default_rng(0)
    starts = [lik.x0, *rng.uniform(-2.0, 2.0, (20, lik.dim))]
    fit = maximize(lik, starts, log=lambda m: print(f"[ic] {task.model}: {m}", flush=True))
    probe = generic(fit.x)
    moved = [j for j in range(lik.dim)
             if np.any(np.abs(lik.class_probs(np.where(np.arange(lik.dim) == j, probe + 0.37, probe))
                              - lik.class_probs(probe)) > 1e-6)]
    k = len(moved)
    ev = flat_log_evidence(lik, start=fit.x, n_samples=n_samples, seed=0)
    return {"model": task.model, "n": lik.n_trials, "n_units": 0, "per_unit_k": 0,
            "unused_coordinates": lik.dim - k,
            "conditional": {"log_lik": fit.log_lik, "k": k,
                            "jacobian_rank": jacobian_rank(lik.class_probs, probe, moved, rtol=1e-4),
                            "grad_norm": fit.grad_norm, "optimizer": fit.message, "iterations": fit.iterations,
                            "running_off": [lik.coordinate_names[j] for j in np.where(np.abs(fit.x) > 12)[0]],
                            **criteria(fit.log_lik, k, lik.n_trials)},
            "marginal": None, "evidence": _evidence_record(ev, None),
            "inputs": {"responses_sha256": bar["data"]["sha256"]}}


def compare(winner: Dict[str, Any], seed: Dict[str, Any]) -> Dict[str, Any]:
    """Winner minus seed, in nats: log-likelihood, parameters, and AIC/2 and
    BIC/2 signed so that positive prefers the winner."""
    out = {}
    for convention in ("conditional", "marginal"):
        if convention == "marginal" and winner["marginal"] is None and seed["marginal"] is None:
            out[convention] = None
            continue
        w = winner[convention] or winner["conditional"]
        s = seed[convention] or seed["conditional"]
        out[convention] = {"log_lik": w["log_lik"] - s["log_lik"], "k": w["k"] - s["k"],
                           "aic": (s["aic"] - w["aic"]) / 2, "bic": (s["bic"] - w["bic"]) / 2}
    out["log_bf"] = winner["evidence"]["log_evidence"] - seed["evidence"]["log_evidence"]
    return out


def _record_path(work_dir: Path, task: Task) -> Path:
    return work_dir / "records" / task.domain / task.run / f"{task.model}.json"


def _elpd_differences(domain: str, run: str, winner: str):
    """The winner's run (an RSA chain) and Figure 2's ELPD of the winner minus
    each seed's: PSIS-LOO on all runs pooled (subjective randomness), the log
    predictive density on the example sets no run drew (number game), and on
    the pre-registered experiment (reference games)."""
    if domain == "number_game":
        unseen = json.loads(NG_UNSEEN.read_text())
        lpd = {m: v["lpd"] for m, v in unseen["unseen_example_sets"]["models"].items()}
        return unseen["winner_runs"][winner], {m: lpd[winner] - v for m, v in lpd.items() if m != winner}
    data = json.loads((FIGURE_DATA / f"{domain}.json").read_text())
    if domain == "subjective_randomness":
        pooled = data["pooled"]
        vs = pooled["vs_best_seed"]
        return pooled["winner_runs"][winner], {m: vs[winner]["diff"] - v["diff"] for m, v in vs.items() if m != winner}
    rows = [r for r in data["runs"] if r["best_model"] == winner]
    if len(rows) != 1:
        raise RuntimeError(f"{domain}: no single figure-data row for {run}/{winner}")
    vs = rows[0]["vs_best_seed"]
    return rows[0]["run"], {m: vs[winner]["diff"] - v["diff"] for m, v in vs.items() if m != winner}


def summarize(work_dir: Path) -> Dict[str, Any]:
    all_tasks = tasks()
    records = {}
    for t in all_tasks:
        path = _record_path(work_dir, t)
        if not path.exists():
            raise FileNotFoundError(f"no record for {t.domain}/{t.run}/{t.model} at {path}")
        records[(t.domain, t.run, t.model)] = json.loads(path.read_text())
    comparisons = []
    for t in all_tasks:
        if t.role != "winner":
            continue
        run, elpd = _elpd_differences(t.domain, t.run, t.model)
        for s in all_tasks:
            if (s.domain, s.run, s.role) != (t.domain, t.run, "seed"):
                continue
            row = {"domain": t.domain, "run": run, "winner": t.model, "seed": s.model,
                   "elpd": elpd.get(s.model),
                   **compare(records[(t.domain, t.run, t.model)], records[(s.domain, s.run, s.model)])}
            comparisons.append(row)
    return {"note": __doc__.split("\n\n")[0], "records": list(records.values()), "comparisons": comparisons}


@dataclass
class Args:
    mode: Literal["tasks", "run", "summarize"]
    task: Optional[int] = None
    """``--mode run``: the task's index in ``--mode tasks``'s list (a Slurm array index)."""
    work_dir: Path = here() / "data" / "paper" / "work" / "information_criteria"
    sr_fit_dir: Path = here() / "data" / "paper" / "work" / "subjective_randomness"
    """``sr_figure_data.py``'s work directory (``<run>/fit_cache``)."""
    rsa_pooled: Optional[Path] = None
    n_samples: int = 4000
    out: Path = here() / "data" / "paper" / "information_criteria.json"


def main(args: Args) -> None:
    all_tasks = tasks()
    if args.mode == "tasks":
        for i, t in enumerate(all_tasks):
            print(i, t.domain, t.run, t.role, t.model)
        return
    if args.mode == "run":
        if args.task is None:
            raise ValueError("--mode run needs --task")
        task = all_tasks[args.task]
        print(f"[ic] task {args.task}: {task.domain} {task.run} {task.role} {task.model}", flush=True)
        record = run_task(task, work_dir=args.work_dir, sr_fit_dir=args.sr_fit_dir, rsa_pooled=args.rsa_pooled,
                          n_samples=args.n_samples)
        path = _record_path(args.work_dir, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=float) + "\n")
        print(f"[ic] wrote {path}", flush=True)
        return
    summary = summarize(args.work_dir)
    args.out.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"[ic] wrote {args.out}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
