"""The rehearsal's hidden ground truth (docs/auto_open/PLAN.md §9, Phase 3).

A simulated chain needs simulated people (`src.rsa_open.chain.GroundTruth`): a theory at fixed
parameters that answers every paradigm the designer freezes, through that paradigm's link. The
rule is the closed campaign's rehearsals 2 and 3 (`src.rsa.outer.ground_truth`, "coherent",
PI 2026-10-10): a theory that fits the human literature data E about as well as the best ones, so
E (real) and the experiments (simulated) do not come from incompatible processes, and that the
chain does not start with, so it has to be found. By default it is the very model those
rehearsals used (`REHEARSAL_TRUTH`, picked from run 2 by that rule for chain 0, named only in
`REHEARSAL_RECORD`), so the open rehearsal is comparable with theirs.

    uv run python -m src.rsa_open.ground_truth --bar <the chain's bar> --out <private dir> \
        --cache <fit cache> --literature <E: every literature trial, all_trials.csv>

1. the closed model file is ported (`src.rsa_open.port.port_file`) to ``<out>/ground_truth.py``;
2. a truth the bar already holds (by name, or the same ported source) is refused;
3. it must meet the listener and speaker contracts on E's scenes and the plain pool (the
   designer may ask for any of them, and a production paradigm reads the speaker);
4. it is fitted to E alone and must converge; its theory parameters are fixed at their
   posterior means, and every link's shared response process (temperature, lapse) at E's
   posterior means; each link's own parameters (a rating's cutpoints, ...) stay at the link's
   reference values. (The closed rehearsals refitted the truth on the data so far each
   experiment and answered from its posterior-mean probabilities; here the data after E are
   the truth's own, so one fit on E fixes it for the whole chain.)

``<out>/ground_truth.json`` records all of it; `load_ground_truth` reads it back. The output is
private: no agent may see the truth's name, file or parameters (CAMPAIGN_LOG #6).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tyro
from pyprojroot import here

from src.models.model_manifest import read_manifest_entries
from src.rsa.fit import FitSettings
from src.rsa_open.chain import GroundTruth
from src.rsa_open.corpus import Block, block_draws, literature_block
from src.rsa_open.links import LINKS
from src.rsa_open.port import port_file

REHEARSAL_RECORD = here() / "data/rsa/rehearsal2/ground_truth.json"
# The truth is never named under src/, which the theorists can read: its file is the one the closed
# record names (a path on Sherlock, from the repository root on).
SHARED_LINK_PARAMS = ("temperature", "lapse")
TRUTH_FILE = "ground_truth.py"
RECORD_FILE = "ground_truth.json"


class GroundTruthRefused(ValueError):
    """The model cannot be a rehearsal's ground truth (the reason says why)."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _body(source: str) -> str:
    """A ported theory without the port's header line (which names the original's file)."""
    return "".join(line for line in source.splitlines(keepends=True) if not line.startswith("# Ported from "))


def _bar_clash(name: str, source: str, bar: Path) -> Optional[str]:
    for e in read_manifest_entries(bar):
        if e["name"] == name:
            return f"the bar has a theory named {name}"
        if _body((Path(bar) / f"{e['name']}.py").read_text()) == _body(source):
            return f"its port is the bar's {e['name']} (apart from the port's header)"
    return None


def _check_contracts(theory, literature: Block) -> None:
    from src.rsa.context import group_by_shape, unique_contexts
    from src.rsa.loop.novelty import plain_pool
    from src.rsa.model_file import ModelContractViolation, check_contract
    from src.rsa_open.loop.gates import check_speaker_contract

    scenes = unique_contexts(list(literature.contexts))[0]
    try:
        for groups in (group_by_shape(scenes), group_by_shape(plain_pool())):
            check_contract(theory, groups)
            check_speaker_contract(theory, groups)
    except ModelContractViolation as exc:
        raise GroundTruthRefused(f"contract: {exc}") from exc


def _checkout_of(path: Path) -> Path:
    """The repository a file under its data/ belongs to."""
    parts = Path(path).resolve().parts
    if "data" not in parts:
        raise ValueError(f"{path} is not under a repository's data/")
    return Path(*parts[:len(parts) - 1 - parts[::-1].index("data")])


def rehearsal_truth(record: Path = REHEARSAL_RECORD) -> Path:
    """The closed rehearsals' truth file, in the checkout that holds ``record`` (the staged
    harness copy has no data/)."""
    recorded = Path(json.loads(Path(record).read_text())["ground_truth_file"])
    if "data" not in recorded.parts:
        raise ValueError(f"{record}: {recorded} is not under the repository's data/")
    path = _checkout_of(record).joinpath(*recorded.parts[recorded.parts.index("data"):])
    if not path.exists():
        raise FileNotFoundError(f"{path} (from {record}) is missing")
    return path


def make_ground_truth(model: Path, *, bar: Path, out: Path, cache: Path, settings: FitSettings,
                      literature: Block, provenance: Optional[Dict[str, Any]] = None) -> dict:
    """Port, check, fit on ``literature`` and record ``model`` as a ground truth in ``out``
    (once: a finished record of the same model is returned as it is)."""
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop.fitting import loop_fit, write_corpus

    out = Path(out)
    record_path = out / RECORD_FILE
    original = Path(model).read_text()
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["original_sha256"] != _sha(original):
            raise FileExistsError(f"{record_path} records another model ({record['original']}); use a new --out")
        return record
    name = Path(model).stem
    source = port_file(Path(model))
    clash = _bar_clash(name, source, bar)
    if clash:
        raise GroundTruthRefused(f"{name} cannot be the ground truth: {clash} (the chain must have to find it)")
    out.mkdir(parents=True, exist_ok=True)
    path = out / TRUTH_FILE
    path.write_text(source)
    theory = TheoryModel(path, name="ground_truth")
    _check_contracts(theory, literature)
    corpus = write_corpus(out / "literature_corpus.json", [literature])
    fitted = loop_fit(path, "ground_truth", corpus, settings, Path(cache), time_limit_sec=None)
    if not fitted.converged:
        raise GroundTruthRefused(f"{name}'s fit on {literature.name} did not converge: "
                                 + "; ".join(fitted.convergence_problems))
    flat = fitted.flat()
    theory_params = {k: float(np.mean(flat[k])) for k in theory.params}
    on_e = block_draws(flat, theory, literature)  # E's temperature is fixed (corpus.LITERATURE_FIXED)
    shared = {k: float(np.mean(on_e[k])) for k in SHARED_LINK_PARAMS}
    record = {
        "ground_truth": name, "original": str(model), "original_sha256": _sha(original),
        "theory_file": TRUTH_FILE, "theory_sha256": _sha(source),
        "rule": "coherent (the closed rehearsals' rule): fits the literature about as well as the best "
                "models and is not one the chain starts with; fixed at its posterior means on "
                f"{literature.name} ({literature.n} trials)",
        "theory_params": theory_params,
        "link_params": {link: dict(shared) for link in sorted(LINKS)},
        "fit": {"settings": {"num_warmup": settings.num_warmup, "num_samples": settings.num_samples,
                             "num_chains": settings.num_chains, "seed": settings.seed},
                "block": literature.name, "n_trials": literature.n,
                "posterior_sd": {k: float(np.std(v)) for k, v in flat.items()}},
        "bar": str(bar),
        "provenance": provenance or {},
    }
    tmp = record_path.with_name(RECORD_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, record_path)
    return record


def load_ground_truth(out: Path) -> GroundTruth:
    """The recorded ground truth, its theory file checked against the record."""
    record = json.loads((Path(out) / RECORD_FILE).read_text())
    path = Path(out) / record["theory_file"]
    if _sha(path.read_text()) != record["theory_sha256"]:
        raise ValueError(f"{path} is not the theory {Path(out) / RECORD_FILE} records")
    return GroundTruth(path, record["theory_params"], record["link_params"])


@dataclass
class Args:
    bar: Path
    """The chain's bar (a models directory with its manifest): the truth must not be in it."""
    out: Path
    """A private directory (no agent may read it)."""
    cache: Path
    literature: Path
    """E: the closed campaign's existing data, all five literature sources (promote's all_trials.csv on
    Sherlock), not the committed pragmods trials alone."""
    model: Optional[Path] = None
    """A closed-campaign model file (default: the closed rehearsals' coherent truth, `rehearsal_truth`)."""
    record: Path = REHEARSAL_RECORD
    """The closed rehearsals' record, which names their truth (in a checkout with data/)."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0


def main(args: Args) -> dict:
    provenance = {}
    model = args.model or rehearsal_truth(args.record)
    if args.model is None:
        closed = json.loads(Path(args.record).read_text())
        provenance = {k: closed[k] for k in ("ground_truth_key", "rule")}
        provenance["record"] = str(Path(args.record).resolve().relative_to(_checkout_of(args.record)))
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed, dense_mass=True)
    record = make_ground_truth(model, bar=args.bar, out=args.out, cache=args.cache, settings=settings,
                               literature=literature_block(args.literature), provenance=provenance)
    print(json.dumps({k: record[k] for k in ("ground_truth", "theory_params", "rule")}, indent=1))
    return record


if __name__ == "__main__":
    main(tyro.cli(Args))
