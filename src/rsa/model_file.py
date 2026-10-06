"""Loading memo model files, one compiled copy per context shape, and their contract.

A model file is a self-contained ``.py`` file that defines

* ``PARAMS`` — a non-empty dict mapping parameter names to numpyro
  distributions (the priors), and
* ``choice_probs(params, ctx)`` — for ONE trial, the probability that the
  participant chooses each object: an array of shape ``(N_OBJ,)``. ``params``
  maps each name in ``PARAMS`` to a scalar; ``ctx`` is a
  `src.rsa.context.ContextArrays` for that trial.

It writes its memo models over the domains ``OBJ`` (objects) and ``UTT``
(utterances), which it must not define: the loader executes the file once per
context shape ``(N_OBJ, N_UTT)`` with those names already bound to
``jnp.arange(N_OBJ)`` / ``jnp.arange(N_UTT)`` (memo's domains are fixed
globals, and padding contexts to one size makes 0/0 normalisations whose
gradients are NaN — see `src.rsa.context`). The harness ``vmap``s
``choice_probs`` over a shape group's trials.

The contract (`check_contract`), run with no sampling: at several parameter
draws from the priors, on every shape group, the probabilities have shape
``(N_OBJ,)`` per trial, are finite and non-negative and sum to 1, and their
gradient with respect to every parameter is finite. A model that breaks it
raises `ModelContractViolation` with the reason.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Distribution

from src.rsa.context import ShapeGroup

INJECTED_NAMES = ("OBJ", "UTT")
PROBABILITY_SUM_TOLERANCE = 1e-4  # memo computes in float32
NEGATIVE_TOLERANCE = 1e-6
CONTRACT_DRAWS = 4


class ModelContractViolation(ValueError):
    """A model file that does not meet the model contract (a model's failure)."""


def _defined_names(tree: ast.Module) -> set:
    """Names bound at the top level of a module (assignments, defs, imports)."""
    names = set()
    for node in tree.body:
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


class RSAModel:
    """A memo model file, compiled lazily once per context shape."""

    def __init__(self, path: Path, name: Optional[str] = None) -> None:
        self.path = Path(path).resolve()
        self.name = name or self.path.stem
        self.source = self.path.read_text(encoding="utf-8")
        self.sha = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        tree = ast.parse(self.source, filename=str(self.path))
        clash = sorted(set(INJECTED_NAMES) & _defined_names(tree))
        if clash:
            raise ModelContractViolation(
                f"{self.path.name} defines {', '.join(clash)}; the domains OBJ and UTT "
                f"are injected by the loader, sized to each context — remove the definition"
            )
        self._code = compile(self.source, str(self.path), "exec")
        self._modules: Dict[Tuple[int, int], types.ModuleType] = {}
        self._jitted: Dict[Tuple[int, int], Any] = {}

    @property
    def compiled_shapes(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(self._modules))

    def module(self, shape: Tuple[int, int]) -> types.ModuleType:
        """The file executed with OBJ/UTT sized to ``shape`` = (N_OBJ, N_UTT)."""
        if shape not in self._modules:
            n_obj, n_utt = shape
            mod_name = f"rsa_model_{self.name}_{self.sha[:12]}_{n_obj}x{n_utt}"
            module = types.ModuleType(mod_name)
            module.__file__ = str(self.path)
            module.__dict__.update(OBJ=jnp.arange(n_obj), UTT=jnp.arange(n_utt))
            # memo reads its source back with inspect, which needs the module
            # registered under its name.
            sys.modules[mod_name] = module
            exec(self._code, module.__dict__)
            self._modules[shape] = module
        return self._modules[shape]

    def _any_module(self) -> types.ModuleType:
        return self.module(self.compiled_shapes[0] if self._modules else (2, 2))

    @property
    def params(self) -> Dict[str, Distribution]:
        module = self._any_module()
        params = getattr(module, "PARAMS", None)
        if not isinstance(params, Mapping) or not params:
            raise ModelContractViolation(
                f"{self.path.name} must define PARAMS, a non-empty dict of "
                f"parameter name -> numpyro distribution (the priors)"
            )
        bad = [k for k, v in params.items() if not isinstance(v, Distribution)]
        if bad:
            raise ModelContractViolation(
                f"PARAMS entries {bad} in {self.path.name} are not numpyro distributions"
            )
        return dict(params)

    def _choice_probs(self, shape: Tuple[int, int]):
        fn = getattr(self.module(shape), "choice_probs", None)
        if not callable(fn):
            raise ModelContractViolation(
                f"{self.path.name} must define choice_probs(params, ctx) returning the "
                f"probability of choosing each object, shape (N_OBJ,)"
            )
        return fn

    def group_probs(self, params: Mapping[str, Any], group: ShapeGroup) -> jnp.ndarray:
        """(trials, N_OBJ) choice probabilities for one shape group."""
        if group.shape not in self._jitted:
            fn = self._choice_probs(group.shape)
            self._jitted[group.shape] = jax.jit(
                lambda p, arrays: jax.vmap(lambda ctx: fn(p, ctx))(arrays)
            )
        return self._jitted[group.shape](dict(params), group.arrays)


def prior_draws(
    params: Mapping[str, Distribution], n: int, seed: int = 0
) -> list[Dict[str, jnp.ndarray]]:
    keys = jax.random.split(jax.random.PRNGKey(seed), len(params))
    samples = {
        name: dist.sample(key, (n,)) for (name, dist), key in zip(params.items(), keys)
    }
    return [{name: v[i] for name, v in samples.items()} for i in range(n)]


def check_contract(
    model: RSAModel,
    groups: Mapping[Tuple[int, int], ShapeGroup],
    *,
    n_draws: int = CONTRACT_DRAWS,
    seed: int = 0,
) -> None:
    """Raise `ModelContractViolation` unless ``model`` meets the contract on ``groups``."""
    params = model.params
    for draw in prior_draws(params, n_draws, seed):
        for shape, group in groups.items():
            n_trials, n_obj = len(group.indices), shape[0]
            where = f"{model.path.name} on {n_obj}-object contexts at params {_fmt(draw)}"
            p = np.asarray(model.group_probs(draw, group))
            if p.shape != (n_trials, n_obj):
                raise ModelContractViolation(
                    f"{where}: choice_probs must return shape (N_OBJ,) = ({n_obj},) per "
                    f"trial; got {p.shape[1:]} per trial"
                )
            if not np.all(np.isfinite(p)):
                raise ModelContractViolation(f"{where}: probabilities are not all finite")
            if p.min() < -NEGATIVE_TOLERANCE:
                raise ModelContractViolation(
                    f"{where}: negative probability {p.min():.3g}"
                )
            worst = float(np.abs(p.sum(axis=1) - 1.0).max())
            if worst > PROBABILITY_SUM_TOLERANCE:
                raise ModelContractViolation(
                    f"{where}: probabilities over objects must sum to 1 (off by {worst:.3g})"
                )
            grads = jax.grad(
                lambda d: jnp.sum(model.group_probs(d, group) ** 2)
            )(draw)
            bad = sorted(k for k, g in grads.items() if not np.all(np.isfinite(np.asarray(g))))
            if bad:
                raise ModelContractViolation(
                    f"{where}: the gradient with respect to {', '.join(bad)} is not finite "
                    f"(NUTS cannot sample it). Common causes: log(0) — add EPS inside the "
                    f"log; a distribution whose weights are all zero for some value (0/0)"
                )


def _fmt(draw: Mapping[str, Any]) -> str:
    return "{" + ", ".join(f"{k}={float(v):.3g}" for k, v in draw.items()) + "}"


def load_models(models_dir: Path, names: Sequence[str]) -> list[RSAModel]:
    return [RSAModel(Path(models_dir) / f"{name}.py", name=name) for name in names]
