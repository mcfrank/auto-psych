"""A loop's saved results: each theory's out-of-fold scores and pool predictions, on disk, so a
resumed loop reads them instead of recomputing them (`DISTRIBUTED_DESIGN_REVIEW_V3.md`, A1).

Profiled on open chain 1 (job 50929501, 27 live theories): a resume took 74 minutes on one core, of
which recomputing held-out scores was 64.8 and pool predictions 8.9. Both are deterministic given
the cached fits, so saving them changes no result; it is what makes restarting the loop at every
step (the step design) affordable.

Each result is keyed by everything it depends on, so a stale one is never read:
- **out-of-fold scores** (`CorpusCV`, or a fold's failure): the theory file, the corpus, the fold
  assignment, the sampler settings of the full fit and of the folds, and the folds' time limit;
- **pool predictions**: the theory file, the corpus, the full fit's settings and the pool.
Files are ``<loop>/.scores/<name>.<key>.npz`` with a ``.json`` beside, each written under a
temporary name and renamed into place.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.rsa.fit import FitSettings
from src.rsa_open.corpus import CorpusCV

SCORES_FORMAT = 1


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\0")
    return h.hexdigest()[:20]


def folds_digest(folds: Mapping[str, np.ndarray]) -> str:
    return _digest({b: np.asarray(f).astype(int).tolist() for b, f in sorted(folds.items())})


class ScoreStore:
    """Saved scores of one loop (one corpus and one fold assignment)."""

    def __init__(self, root: Path, corpus_path: Path, folds: Mapping[str, np.ndarray], settings: FitSettings,
                 fold_settings: FitSettings, pool_size: int, pool_draws: int) -> None:
        self.root = Path(root)
        self._corpus = _sha_file(corpus_path)
        self._folds = folds_digest(folds)
        self._settings = asdict(settings)
        self._fold_settings = asdict(fold_settings)
        self._pool = (pool_size, pool_draws)

    def cv_key(self, theory_path: Path, fold_limit_sec: Optional[float]) -> str:
        return _digest(f"open-cv{SCORES_FORMAT}", _sha_file(theory_path), self._corpus, self._folds, self._settings,
                       self._fold_settings, None if fold_limit_sec is None else round(float(fold_limit_sec), 3))

    def preds_key(self, theory_path: Path) -> str:
        return _digest(f"open-preds{SCORES_FORMAT}", _sha_file(theory_path), self._corpus, self._settings, self._pool)

    def _paths(self, name: str, key: str) -> Tuple[Path, Path]:
        return self.root / f"{name}.{key}.npz", self.root / f"{name}.{key}.json"

    def _write(self, name: str, key: str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        npz, js = self._paths(name, key)
        tmp = npz.with_name(f".{npz.name}.tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp, npz)
        tmp = js.with_name(f".{js.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(dict(meta)))
        os.replace(tmp, js)  # the .json last: it marks the result complete

    def _read(self, name: str, key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        npz, js = self._paths(name, key)
        if not js.exists():
            return None
        meta = json.loads(js.read_text())
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, meta

    # ---------- out-of-fold scores ----------
    def save_cv(self, name: str, theory_path: Path, fold_limit_sec: Optional[float], cv: Optional[CorpusCV],
                problem: Optional[str]) -> None:
        arrays = {} if cv is None else {f"block:{b}": v for b, v in cv.pointwise.items()}
        self._write(name, self.cv_key(theory_path, fold_limit_sec), arrays,
                    {"name": name, "scored": cv is not None, "converged": None if cv is None else bool(cv.converged),
                     "problem": problem, "blocks": [] if cv is None else list(cv.pointwise)})

    def load_cv(self, name: str, theory_path: Path,
                fold_limit_sec: Optional[float]) -> Optional[Tuple[Optional[CorpusCV], Optional[str]]]:
        """``(the scores or None for a failed fold, the recorded problem)``, or None when nothing is saved."""
        got = self._read(name, self.cv_key(theory_path, fold_limit_sec))
        if got is None:
            return None
        arrays, meta = got
        if not meta["scored"]:
            return None, meta["problem"]
        return CorpusCV({b: arrays[f"block:{b}"] for b in meta["blocks"]}, bool(meta["converged"])), meta["problem"]

    # ---------- pool predictions ----------
    def save_preds(self, name: str, theory_path: Path, preds: np.ndarray) -> None:
        self._write(name, self.preds_key(theory_path), {"preds": preds}, {"name": name})

    def load_preds(self, name: str, theory_path: Path) -> Optional[np.ndarray]:
        got = self._read(name, self.preds_key(theory_path))
        return None if got is None else got[0]["preds"]
