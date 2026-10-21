"""Admission of a theorist's theory to the open loop (the closed loop's `src.rsa.loop.gates`, for
theories fitted to every paradigm of the chain).

A candidate directory holds ``candidate.py`` (a theory: ``PARAMS``, ``listener(params, ctx)``
and optionally ``speaker(params, ctx)``, `src.rsa_open.fit.TheoryModel`) and ``hypothesis.md``.
It is admitted only if, in order:

1. the code gate passes (`src.rsa.loop.code_gate`);
2. it is a theory, not a model of one paradigm: it defines ``listener`` and no
   ``choice_probs``, and none of its parameters is a link's (temperature, lapse and the
   links' own: the response process belongs to the paradigm, `src.rsa_open.links`);
3. it loads and meets the **listener contract** (`src.rsa.model_file.check_contract`) and the
   **speaker contract** (`check_speaker_contract`: every target's distribution over the
   scene's real messages, the sink never said, finite gradients) on every scene of the corpus
   and of the novelty pool;
4. its fit to the corpus (every block, `src.rsa_open.loop.fitting.loop_fit`) finishes within
   the time limit, gives no observed response probability zero and converges (after the one
   refit);
5. it is novel: its posterior-mean predictions on the novelty pool, listener and speaker
   both, are at least the threshold (RMSE) from every admitted theory's.

Every refusal carries a reason written for the agent that will repair it; infrastructure
failures raise.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from src.rsa.context import Context, group_by_shape, unique_contexts
from src.rsa.fit import FitSettings, class_probs, draw_batches
from src.rsa.loop.code_gate import code_problems
from src.rsa.loop.fitting import ModelFailure, is_model_failure
from src.rsa.loop.gates import read_candidate
from src.rsa.loop.novelty import closest
from src.rsa.model_file import ModelContractViolation, check_contract, prior_draws
from src.rsa_open.corpus import Block, CorpusFit
from src.rsa_open.fit import TheoryModel
from src.rsa_open.loop.fitting import loop_fit

MAX_POOL_DRAWS = 100
SPEAKER_SUM_TOLERANCE = 1e-4


@dataclass
class Admission:
    admitted: bool
    reason: str
    fit: Optional[CorpusFit] = None
    pool_preds: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GateConfig:
    corpus_path: Path
    cache_dir: Path
    settings: FitSettings
    novelty_threshold: float
    time_limit_sec: Optional[float]


def _refuse(reason: str, **kw) -> Admission:
    return Admission(False, reason, **kw)


def link_parameter_names() -> set:
    from src.rsa_open.links import LINKS, likelihood

    return {k for name in LINKS for k in likelihood(name, scale_points=5, total_points=100).params}


def theory_shape_problems(source: str) -> list:
    """Why a file is not a paradigm-free theory (empty when it is one)."""
    tree = ast.parse(source)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems = []
    if "listener" not in defined:
        problems.append("it defines no listener(params, ctx)")
    if "choice_probs" in defined:
        problems.append("it defines choice_probs: a theory gives the listener's posterior (listener) and, if "
                        "it likes, the speaker (speaker); the paradigm's link turns them into responses")
    return problems


def check_speaker_contract(theory: TheoryModel, groups, *, n_draws: int = 2, seed: int = 0) -> None:
    """Raise `ModelContractViolation` unless the theory's speaker (its own or the derived one) is,
    for every target, a distribution over the scene's real messages with finite gradients."""
    for draw in prior_draws(theory.params, n_draws, seed) if theory.params else [{}]:
        for shape, group in groups.items():
            n_obj, n_utt = shape
            where = f"{theory.path.name}'s speaker on {n_obj}-object scenes"
            s = np.asarray(theory.group_speaker(draw, group))
            if s.shape != (len(group.indices), n_utt, n_obj):
                raise ModelContractViolation(f"{where}: speaker must return shape (N_UTT, N_OBJ) = ({n_utt}, {n_obj}) "
                                             f"per trial; got {s.shape[1:]}")
            if not np.all(np.isfinite(s)) or s.min() < -1e-6:
                raise ModelContractViolation(f"{where}: probabilities are not all finite and non-negative")
            sink = np.asarray(group.arrays.is_sink) > 0
            if np.any(s[np.broadcast_to(sink[..., None], s.shape)] > 1e-6):
                raise ModelContractViolation(f"{where}: the sink ('no word applies') must have probability 0")
            worst = float(np.abs(s.sum(axis=1) - 1.0).max())
            if worst > SPEAKER_SUM_TOLERANCE:
                raise ModelContractViolation(f"{where}: each target's probabilities over messages must sum to 1 "
                                             f"(off by {worst:.3g})")
            if theory.params:
                grads = jax.grad(lambda d: jnp.sum(theory.group_speaker(d, group) ** 2))(draw)
                bad = sorted(k for k, g in grads.items() if not np.all(np.isfinite(np.asarray(g))))
                if bad:
                    raise ModelContractViolation(f"{where}: the gradient with respect to {', '.join(bad)} is not finite")


def pool_predictions(theory: TheoryModel, fitted: CorpusFit, pool: Sequence[Context],
                     max_draws: int = MAX_POOL_DRAWS) -> np.ndarray:
    """Posterior-mean listener class probabilities and speaker probabilities over the pool,
    flattened; raises ValueError when they are not finite."""
    flat = fitted.flat()
    params = {k: flat[k] for k in theory.params}
    n = len(next(iter(flat.values())))
    take = np.unique(np.linspace(0, n - 1, min(n, max_draws)).round().astype(int))
    parts = []
    for group in group_by_shape(pool).values():
        if params:
            acc_l, acc_s = 0.0, 0.0
            for k, batch in draw_batches(params, take):
                acc_l = acc_l + np.asarray(class_probs(theory.draws_probs(batch, group), group.classes)[:k],
                                           dtype=np.float64).sum(axis=0)
                acc_s = acc_s + np.asarray(theory.draws_speaker(batch, group)[:k], dtype=np.float64).sum(axis=0)
            listener, speaker = acc_l / len(take), acc_s / len(take)
        else:
            listener = np.asarray(class_probs(theory.group_probs({}, group)[None], group.classes)[0])
            speaker = np.asarray(theory.group_speaker({}, group))
        parts += [listener.reshape(-1), speaker.reshape(-1)]
    preds = np.concatenate(parts)
    if not np.all(np.isfinite(preds)):
        raise ValueError(f"{theory.name}'s predictions are not finite on some novelty-pool scenes")
    return preds


def admit(candidate_dir: Path, name: str, *, cfg: GateConfig, blocks: Sequence[Block], pool: Sequence[Context],
          admitted_preds: Mapping[str, np.ndarray]) -> Admission:
    candidate_dir = Path(candidate_dir)
    missing = read_candidate(candidate_dir)
    if missing:
        return _refuse(missing)
    path = candidate_dir / "candidate.py"
    source = path.read_text(encoding="utf-8")
    problems = code_problems(source)
    if problems:
        return _refuse("the code gate refused candidate.py: " + ", ".join(problems) +
                       ". Theory files may import jax, memo, numpyro, numpy, a few stdlib modules and "
                       "src.rsa.memo_kit only, and may not read files or reach the interpreter.")
    try:
        shape = theory_shape_problems(source)
    except SyntaxError as exc:
        return _refuse(f"candidate.py is not valid Python: {exc}")
    if shape:
        return _refuse("candidate.py is not a theory: " + "; ".join(shape))
    try:
        theory = TheoryModel(path, name=name)
        clash = sorted(set(theory.params) & link_parameter_names())
        if clash:
            return _refuse(f"its parameters {clash} belong to the links (each paradigm's response process: "
                           f"a decision temperature, a lapse, a rating's cutpoints); rename or remove them")
        scenes = unique_contexts([c for b in blocks for c in b.contexts])[0]
        for groups in (group_by_shape(scenes), group_by_shape(pool)):
            check_contract(theory, groups)
            check_speaker_contract(theory, groups)
    except ModelContractViolation as exc:
        return _refuse(f"contract: {exc}")
    except Exception as exc:  # memo compile errors, errors in the theory's own code
        if is_model_failure(exc, path):
            return _refuse(f"the theory failed to load or run: {type(exc).__name__}: {exc}")
        raise
    try:
        fitted = loop_fit(path, name, cfg.corpus_path, cfg.settings, cfg.cache_dir, time_limit_sec=cfg.time_limit_sec)
    except ModelFailure as exc:
        return _refuse(f"fitting failed: {exc}")
    if not fitted.converged:
        return _refuse("the fit did not converge even at target_accept 0.95: " + "; ".join(fitted.convergence_problems)
                       + ". Reparameterise (sample on an unconstrained scale, tighten priors) rather than relying "
                       "on smaller steps.", fit=fitted)
    try:
        preds = pool_predictions(theory, fitted, pool)
    except ValueError as exc:
        return _refuse(str(exc), fit=fitted)
    nearest, dist = closest(preds, admitted_preds)
    if dist < cfg.novelty_threshold:
        return _refuse(f"not novel: its listener and speaker predictions are within RMSE {dist:.4f} of {nearest}'s "
                       f"(threshold {cfg.novelty_threshold}) over the novelty pool of {len(pool)} scenes. A "
                       f"re-parameterisation of an existing mechanism is not a new hypothesis; change what the "
                       f"listener or speaker computes.", fit=fitted, pool_preds=preds)
    return Admission(True, "admitted", fit=fitted, pool_preds=preds)
