"""Reference-game contexts and their encoding for memo models.

A context is what one trial shows a listener: a few objects (referents), each
a set of features, and either the one feature word a speaker said (an
*utterance* query) or no informative word (a *prior* query: "mumble", "which
will he pick next?").

Models never see padding. memo enumerates over fixed domains, and padding a
small context out to a common size makes some distribution a 0/0 — harmless in
memo's forward pass (it returns zeros) but NaN in its gradient, which stops
NUTS. So trials are grouped by shape (`group_by_shape`) and each group is
evaluated by a copy of the model compiled with exact-size domains `OBJ` and
`UTT` (`src.rsa.model_file`).

The utterances of a context are the features true of at least one object, in
feature order, plus — when some object has none of them — a final **sink**
utterance true of exactly those objects ("no word applies"). Without it a
speaker who means a featureless object (the plain face of the pragmods simple
game) has no true word, again a 0/0. The sink is never observed, and it adds
no probability to any object a real word is true of, so L0/S1/L1 values on
real words are the textbook ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

SINK = "<sink>"


class ContextArrays(NamedTuple):
    """One trial (or, stacked, a group of same-shape trials) as JAX arrays.

    Shapes are for one trial; a group adds a leading trial axis.
    """

    lex: jnp.ndarray  # (N_UTT, N_OBJ) 1.0 where utterance u is true of object r
    is_sink: jnp.ndarray  # (N_UTT,) 1.0 for the sink utterance
    feature_count: jnp.ndarray  # (N_OBJ,) number of context words true of each object
    utterance: jnp.ndarray  # () int32 utterance index heard (0 for a prior query)
    is_prior: jnp.ndarray  # () 1.0 for a prior query (no informative word)
    familiarization: jnp.ndarray  # (N_OBJ,) base rate seen in familiarization, 0..1
    has_familiarization: jnp.ndarray  # () 1.0 when there was a familiarization phase
    grayscale: jnp.ndarray  # (N_OBJ,) 1.0 for an object shown in grayscale
    valence: jnp.ndarray  # () +1 "favorite" framing, -1 "least favorite", 0 neutral


@dataclass(frozen=True)
class Context:
    """One trial's display. ``utterance`` indexes ``feature_names`` (None: prior query)."""

    objects: Tuple[Tuple[int, ...], ...]
    feature_names: Tuple[str, ...]
    utterance: Optional[int]
    item: str = ""
    familiarization: Optional[Tuple[float, ...]] = None
    grayscale: Optional[Tuple[int, ...]] = None
    valence: int = 0

    def __post_init__(self) -> None:
        objects = tuple(tuple(int(v) for v in row) for row in self.objects)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not objects:
            raise ValueError("a context needs at least one object")
        n_features = len(self.feature_names)
        if any(len(row) != n_features for row in objects):
            raise ValueError(
                f"objects must be rectangular with one entry per feature "
                f"({n_features}): {objects}"
            )
        if any(v not in (0, 1) for row in objects for v in row):
            raise ValueError(f"object features must be 0/1: {objects}")
        if self.utterance is not None:
            if not 0 <= self.utterance < n_features:
                raise ValueError(
                    f"utterance {self.utterance} is not a feature index (0..{n_features - 1})"
                )
            if not any(row[self.utterance] for row in objects):
                raise ValueError(
                    f"utterance {self.feature_names[self.utterance]!r} is true of no object"
                )
        for name in ("familiarization", "grayscale"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(value)
                object.__setattr__(self, name, value)
                if len(value) != len(objects):
                    raise ValueError(f"{name} needs one per object: {value}")
        if self.familiarization is not None and not all(
            0.0 <= f <= 1.0 for f in self.familiarization
        ):
            raise ValueError(f"familiarization rates must be in [0, 1]: {self.familiarization}")
        if self.grayscale is not None and any(g not in (0, 1) for g in self.grayscale):
            raise ValueError(f"grayscale must be 0/1 per object: {self.grayscale}")
        if self.valence not in (-1, 0, 1):
            raise ValueError(f"valence must be -1, 0 or 1: {self.valence}")

    @property
    def present_features(self) -> Tuple[int, ...]:
        """Feature indices true of at least one object: the context's words."""
        return tuple(
            f for f in range(len(self.feature_names)) if any(row[f] for row in self.objects)
        )

    @property
    def needs_sink(self) -> bool:
        return any(not any(row) for row in self.objects)

    @property
    def utterance_names(self) -> Tuple[str, ...]:
        names = tuple(self.feature_names[f] for f in self.present_features)
        return names + ((SINK,) if self.needs_sink else ())

    @property
    def shape(self) -> Tuple[int, int]:
        """(N_OBJ, N_UTT): the domain sizes a model is compiled for."""
        return (len(self.objects), len(self.utterance_names))

    def lexicon(self) -> np.ndarray:
        """(N_UTT, N_OBJ) truth table of utterances over objects, sink last."""
        rows = [[row[f] for row in self.objects] for f in self.present_features]
        if self.needs_sink:
            rows.append([int(not any(row)) for row in self.objects])
        return np.asarray(rows, dtype=np.float32)

    def arrays(self) -> ContextArrays:
        n_obj, n_utt = self.shape
        lex = self.lexicon()
        is_sink = np.zeros(n_utt, dtype=np.float32)
        if self.needs_sink:
            is_sink[-1] = 1.0
        if self.utterance is None:
            utterance, is_prior = 0, 1.0
        else:
            utterance, is_prior = self.present_features.index(self.utterance), 0.0
        familiarization = (
            np.zeros(n_obj) if self.familiarization is None else np.asarray(self.familiarization)
        )
        grayscale = np.zeros(n_obj) if self.grayscale is None else np.asarray(self.grayscale)
        return ContextArrays(
            lex=jnp.asarray(lex),
            is_sink=jnp.asarray(is_sink),
            feature_count=jnp.asarray((lex * (1 - is_sink)[:, None]).sum(axis=0)),
            utterance=jnp.asarray(utterance, dtype=jnp.int32),
            is_prior=jnp.asarray(is_prior, dtype=jnp.float32),
            familiarization=jnp.asarray(familiarization, dtype=jnp.float32),
            has_familiarization=jnp.asarray(
                float(self.familiarization is not None), dtype=jnp.float32
            ),
            grayscale=jnp.asarray(grayscale, dtype=jnp.float32),
            valence=jnp.asarray(float(self.valence), dtype=jnp.float32),
        )


@dataclass(frozen=True)
class ShapeGroup:
    """Same-shape trials: their positions in the input and their stacked arrays."""

    shape: Tuple[int, int]
    indices: np.ndarray
    arrays: ContextArrays


def stack(arrays: Sequence[ContextArrays]) -> ContextArrays:
    return ContextArrays(*(jnp.stack(field) for field in zip(*arrays)))


def group_by_shape(contexts: Sequence[Context]) -> Dict[Tuple[int, int], ShapeGroup]:
    """Group contexts by (N_OBJ, N_UTT), keeping input order within a group."""
    by_shape: Dict[Tuple[int, int], List[int]] = {}
    for i, ctx in enumerate(contexts):
        by_shape.setdefault(ctx.shape, []).append(i)
    return {
        shape: ShapeGroup(
            shape=shape,
            indices=np.asarray(idx),
            arrays=stack([contexts[i].arrays() for i in idx]),
        )
        for shape, idx in sorted(by_shape.items())
    }
