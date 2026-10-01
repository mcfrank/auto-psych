"""Re-prune a finished experiment's model loop at another dse multiplier.

Pruning runs once, at the end of an experiment's model loop
(``pymc_orchestrator.end_experiment``), and nothing before it depends on the
multiplier: the rounds, the admissions and every fit are the same whatever it
is. So redoing that last step at another multiplier gives exactly the model
set a loop run with it would have carried, without rerunning the loop. Every
model the step pruned is still in ``model_loop/models/pruned/`` and its fit in
``model_loop/.fit_cache/``.

The step's ledger lines (context ``<experiment> end of experiment``, the last
the loop writes) and its note on the last history step are removed, the
models it retired go back into the zoo in the zoo's order, and
``end_experiment`` runs again at the new multiplier, followed by the live-set
export and the stage's completion (registry and export record). The change is
recorded in ``model_loop/repruned.json``. At the same multiplier the result is
the experiment as it was.

Refused when the next experiment's directory exists (it may already have
started from the old export: move it aside if it ran nothing), when the stage
did not finish, and when any fit the comparison needs is not in the cache
(``fit_kwargs`` must be the loop's own sampler settings; nothing is sampled).

Written for the October 2026 live run (user decision 2026-09-30): pruning at
2·dse_clustered left experiment 1 a single model, which gives the next
design nothing to discriminate; the series moved to 4.

    python -m src.pipelines.outer_loop.reprune --experiment-dir <run>/<project>/experiment1 \\
        --dse-multiplier 4 --draws 3000 --tune 2000 --chains 4 --target-accept 0.8
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import tyro

from src.models.model_manifest import read_manifest_entries
from src.models.pymc_inference import cached_fit_path, fit_fingerprint, resolve_fit_settings
from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
    LedgerEntry,
)
from src.pipelines.inner_loop.model_zoo import _write_manifest
from src.pipelines.inner_loop.pymc_orchestrator import end_experiment
from src.pipelines.inner_loop.scoring import (
    DEFAULT_COMPLEXITY_PRIOR_CONST,
    _compare,
    _score,
)
from src.pipelines.outer_loop.model_loop_runner import (
    MODEL_LOOP_INPUT_DIRNAME,
    _experiment_number,
    _export_inner_loop_models,
    finish_model_loop_stage,
)
from src.pipelines.outer_loop.orchestrator_validators import (
    EXPORT_RECORD_FILENAME,
    validate_cc_output,
)
from src.runtime.atomic_files import write_text_atomically

REPRUNE_RECORD_FILENAME = "repruned.json"


def reprune_experiment(
    exp_dir: Path,
    *,
    dse_multiplier: float,
    fit_kwargs: Dict[str, Any],
    note: str = "",
) -> Dict[str, Any]:
    """Redo ``exp_dir``'s end-of-experiment pruning at ``dse_multiplier``.

    Returns what changed: the models the old step retired, the new step's
    retirements and the live set it exports.
    """
    exp_dir = Path(exp_dir)
    loop_dir = exp_dir / "model_loop"
    models_dir = loop_dir / "models"
    pruned_dir = models_dir / "pruned"
    responses = loop_dir / "responses.csv"
    cache_dir = loop_dir / ".fit_cache"
    ledger_path = loop_dir / LEDGER_FILENAME
    history_path = loop_dir / "history.json"

    ok, message = validate_cc_output("5_model_loop", exp_dir)
    if not ok:
        raise RuntimeError(
            f"{exp_dir}'s model loop did not finish ({message}); only a finished "
            "stage is re-pruned."
        )
    next_dir = exp_dir.parent / f"experiment{_experiment_number(exp_dir) + 1}"
    if next_dir.exists():
        raise RuntimeError(
            f"{next_dir} exists: the next experiment may already have started "
            f"from {exp_dir.name}'s export. Move it aside if it ran nothing (no "
            "deployment, no data), then re-prune."
        )

    # The step's retirements: the ledger's last lines, all of one context.
    end_context = f"{exp_dir.name} end of experiment"
    lines = [
        line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    entries = [LedgerEntry.from_json(line, source=ledger_path) for line in lines]
    n_end = 0
    while n_end < len(entries) and entries[-1 - n_end].context == end_context:
        n_end += 1
    retired_entries = entries[len(entries) - n_end :]
    if any(e.context == end_context for e in entries[: len(entries) - n_end]):
        raise RuntimeError(
            f"{ledger_path} has '{end_context}' lines that are not its last: "
            "the experiment did not end the way the loop ends one."
        )
    if any(e.outcome != "pruned" for e in retired_entries):
        raise RuntimeError(f"Unexpected end-of-experiment ledger lines in {ledger_path}.")
    retired = [e.name for e in retired_entries]
    live_entries = read_manifest_entries(models_dir)
    live = [e["name"] for e in live_entries]
    for name in retired:
        if name in live or not (pruned_dir / f"{name}.py").exists():
            raise RuntimeError(
                f"{name!r} is recorded as pruned at the end of {exp_dir.name} but is "
                f"not in {pruned_dir} (or is live): the zoo is not as the step left it."
            )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if sorted(history[-1].get("retired_at_experiment_end", [])) != sorted(retired):
        raise RuntimeError(
            f"{history_path}'s last step does not record the retirements the "
            f"ledger does ({retired})."
        )

    # Nothing may be sampled: every fit the comparison needs is in the cache.
    for name, directory in [(n, models_dir) for n in live] + [(n, pruned_dir) for n in retired]:
        settings = resolve_fit_settings(name, directory, fit_kwargs)
        fingerprint = fit_fingerprint(name, directory, responses, settings)
        if not cached_fit_path(cache_dir, name, fingerprint).exists():
            raise RuntimeError(
                f"No cached fit of {name!r} at {cache_dir} for {fit_kwargs}: pass the "
                "loop's own sampler settings (re-pruning samples nothing)."
            )

    # Back to the zoo as the step found it: its files, its manifest in zoo
    # order (the input set, then admissions in ledger order) and its ledger.
    rationales = {e["name"]: e["rationale"] for e in live_entries}
    input_entries = read_manifest_entries(exp_dir / MODEL_LOOP_INPUT_DIRNAME)
    for entry in input_entries:
        rationales.setdefault(entry["name"], entry.get("rationale", "") or "Seed model.")
    for name in retired:
        hypothesis_file = pruned_dir / f"{name}.hypothesis.md"
        if name not in rationales:
            rationales[name] = hypothesis_file.read_text(encoding="utf-8").strip()
    order: List[str] = [e["name"] for e in input_entries] + [
        e.name
        for e in entries
        if e.outcome == "admitted" and e.context.startswith(f"{exp_dir.name} ")
    ]
    zoo = set(live) | set(retired)
    restored_order = list(dict.fromkeys(n for n in order if n in zoo))
    missing = zoo - set(restored_order)
    if missing:
        raise RuntimeError(f"Cannot place {sorted(missing)} in the zoo's order.")

    (loop_dir / EXPORT_RECORD_FILENAME).unlink()  # incomplete until finished again
    for name in retired:
        for suffix in (".py", ".hypothesis.md"):
            source = pruned_dir / f"{name}{suffix}"
            if source.exists():
                shutil.move(str(source), str(models_dir / f"{name}{suffix}"))
    _write_manifest(
        models_dir, [{"name": n, "rationale": rationales[n]} for n in restored_order]
    )
    kept_lines = lines[: len(lines) - n_end]
    write_text_atomically(ledger_path, "".join(line + "\n" for line in kept_lines))
    history[-1].pop("retired_at_experiment_end", None)
    write_text_atomically(history_path, json.dumps(history, indent=2))

    # The loop's own end-of-experiment step, then the stage's export.
    posterior = _score(
        responses, models_dir, DEFAULT_COMPLEXITY_PRIOR_CONST, cache_dir, fit_kwargs
    )
    comparison = _compare(responses, models_dir, cache_dir, fit_kwargs)
    result = end_experiment(
        loop_dir,
        models_dir,
        responses,
        history=history,
        posterior=posterior,
        comparison=comparison,
        ledger=HypothesisLedger(ledger_path),
        ledger_context=exp_dir.name,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        prune_dse_multiplier=dse_multiplier,
        complexity_prior_const=DEFAULT_COMPLEXITY_PRIOR_CONST,
    )
    _export_inner_loop_models(exp_dir, loop_dir, best_model=result["best_model"])
    finish_model_loop_stage(exp_dir)

    now_retired = json.loads(history_path.read_text(encoding="utf-8"))[-1].get(
        "retired_at_experiment_end", []
    )
    summary = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dse_multiplier": dse_multiplier,
        "fit_kwargs": fit_kwargs,
        "previously_retired": retired,
        "retired": now_retired,
        "restored": [n for n in retired if n not in now_retired],
        "live": [e["name"] for e in read_manifest_entries(exp_dir / "cognitive_models")],
        "best_model": result["best_model"],
        "note": note,
    }
    record_path = loop_dir / REPRUNE_RECORD_FILENAME
    records = (
        json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else []
    )
    write_text_atomically(record_path, json.dumps(records + [summary], indent=2) + "\n")
    print(
        f"  [reprune] {exp_dir.name} at {dse_multiplier}·clustered dse: restored "
        f"{summary['restored']}; live set {summary['live']}",
        flush=True,
    )
    return summary


@dataclass
class Args:
    """Redo a finished experiment's end-of-experiment pruning at another multiplier."""

    experiment_dir: Path
    """The experiment directory (``<run>/<project>/experimentN``)."""
    dse_multiplier: float
    """Prune models with elpd_diff > this · dse_clustered against the best trusted model."""
    draws: int
    """The loop's MCMC draws per chain (its fits are looked up, never sampled)."""
    tune: int
    """The loop's tuning steps per chain."""
    chains: int
    """The loop's chains."""
    target_accept: Optional[float] = None
    """The loop's target_accept, if the run set one."""
    note: str = ""
    """Why, recorded in model_loop/repruned.json."""


def main(args: Args) -> None:
    fit_kwargs: Dict[str, Any] = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    if args.target_accept is not None:
        fit_kwargs["target_accept"] = args.target_accept
    reprune_experiment(
        args.experiment_dir,
        dse_multiplier=args.dse_multiplier,
        fit_kwargs=fit_kwargs,
        note=args.note,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
