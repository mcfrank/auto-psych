"""The model loop's state after its last completed round, so a stopped loop
resumes with the next round instead of starting over.

Saved after every round: the model set, the ledger, the history and the run's
agent notes. A restore puts them back and removes the working dirs of the
rounds that follow. ``checkpoint_before_round`` builds one for a finished loop.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.model_zoo import _manifest_entries, _write_manifest
from src.runtime.atomic_files import (
    copy_directory_atomically,
    replace_directory,
    write_text_atomically,
)

CHECKPOINT_DIRNAME = "checkpoint"
_STATE_FILES = (LEDGER_FILENAME, "history.json")


def save_round_checkpoint(
    results_dir: Path, notes_dir: Optional[Path], *, next_round: int, rounds_abandoned: int
) -> None:
    results_dir = Path(results_dir)

    def build(staging: Path) -> None:
        shutil.copytree(results_dir / "models", staging / "models")
        for name in _STATE_FILES:
            shutil.copyfile(results_dir / name, staging / name)
        if notes_dir is not None and Path(notes_dir).is_dir():
            shutil.copytree(notes_dir, staging / "agent_notes")
        state = {"next_round": next_round, "rounds_abandoned": rounds_abandoned}
        (staging / "state.json").write_text(json.dumps(state), encoding="utf-8")

    replace_directory(results_dir / CHECKPOINT_DIRNAME, build)


def restore_round_checkpoint(results_dir: Path, notes_dir: Optional[Path]) -> Optional[Dict[str, int]]:
    """Put the loop back to its checkpoint and return its state; None when it has none."""
    results_dir = Path(results_dir)
    checkpoint = results_dir / CHECKPOINT_DIRNAME
    if not checkpoint.is_dir():
        return None
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    copy_directory_atomically(checkpoint / "models", results_dir / "models")
    for name in _STATE_FILES:
        write_text_atomically(results_dir / name, (checkpoint / name).read_text(encoding="utf-8"))
    if notes_dir is not None:
        if (checkpoint / "agent_notes").is_dir():
            copy_directory_atomically(checkpoint / "agent_notes", notes_dir)
        elif Path(notes_dir).exists():
            shutil.rmtree(notes_dir)
    for round_dir in results_dir.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)(_retry_\d+)?", round_dir.name)
        if match and int(match.group(1)) >= state["next_round"]:
            shutil.rmtree(round_dir)
    return state


def checkpoint_before_round(
    results_dir: Path, next_round: int, notes_dir: Optional[Path], *, ledger_context: str
) -> None:
    """Checkpoint a finished loop as it stood before round ``next_round``.

    For a loop whose rounds from ``next_round`` on were abandoned (which
    changes nothing) and that then ended the experiment: the end-of-experiment
    pruning is undone, and the ledger lines of those rounds and of the end are
    dropped. ``notes_dir`` must already hold the notes of that point.
    """
    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    history = json.loads((results_dir / "history.json").read_text(encoding="utf-8"))
    if any(s["iteration"] is not None and s["iteration"] >= next_round for s in history):
        raise ValueError(f"{results_dir}: a round from {next_round} on completed")
    history[-1].pop("retired_at_experiment_end", None)

    end = f"{ledger_context} end of experiment".strip()
    round_prefix = f"{ledger_context} round ".lstrip()
    lines = (results_dir / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    def undone(context: str) -> bool:
        return context == end or (
            context.startswith(round_prefix)
            and int(context[len(round_prefix):].split(" ")[0]) >= next_round
        )

    cut = next(i for i, e in enumerate(entries) if undone(e["context"]))
    if not all(undone(e["context"]) for e in entries[cut:]):
        raise ValueError(f"{results_dir}: the ledger continues past round {next_round - 1}")

    restored = {e["name"]: e["hypothesis"] for e in entries[cut:] if e["outcome"] == "pruned"}
    for name in restored:
        for suffix in (".py", ".hypothesis.md"):
            if (models_dir / "pruned" / f"{name}{suffix}").exists():
                shutil.move(str(models_dir / "pruned" / f"{name}{suffix}"), str(models_dir))
    by_name = {e["name"]: e for e in _manifest_entries(models_dir)}
    by_name |= {name: {"name": name, "rationale": why} for name, why in restored.items()}
    order = list(history[-1]["posteriors"])  # the set as last scored, before the end
    if set(order) != set(by_name):
        raise ValueError(f"{results_dir}: restored {sorted(by_name)} != last scored {sorted(order)}")
    _write_manifest(models_dir, [by_name[name] for name in order])

    write_text_atomically(results_dir / LEDGER_FILENAME, "".join(f"{line}\n" for line in lines[:cut]))
    write_text_atomically(results_dir / "history.json", json.dumps(history, indent=2))
    save_round_checkpoint(
        results_dir,
        notes_dir,
        next_round=next_round,
        rounds_abandoned=sum(
            e["outcome"] == "round_abandoned" and e["context"].startswith(round_prefix)
            for e in entries[:cut]
        ),
    )
