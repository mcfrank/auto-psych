"""CLI: resync the outer loop's live seed pool from the recovery registry.

The registry at ``src/subjective_randomness/pymc_model_families/`` is the single
source of truth for the active seed set. The outer loop's live seed pool at
``src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`` is a
verbatim mirror (it is copied into experiment 1 and shown to coding agents as
worked examples), with one documented exception, ``SEED_SOURCES``: the pool's
``motif_stack`` seed is a copy of the registry's ``motif_stack_softmax.py``.
Edit the registry, then run this to propagate the change;
``tests/test_model_manifest.py`` fails loudly if the two ever diverge.

Usage:
    uv run python scripts/subjective_randomness/sync_seed_models.py [--check]
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.models.model_manifest import MANIFEST_FILENAME, read_manifest_names  # noqa: E402

REGISTRY_DIR = here() / "src" / "subjective_randomness" / "pymc_model_families"
SEED_POOL_DIR = (
    here()
    / "src"
    / "pipelines"
    / "outer_loop"
    / "projects"
    / "subjective_randomness"
    / "seed_models"
)

# The one seed whose file is not the registry model of the same name (user
# decision 2026-09-27). The registry's motif_stack.py is Griffiths et al.'s
# Viterbi (max-path, max-method) automaton and stays the ground truth that
# generates data; the loop is seeded with its marginalising rewrite, which fits
# 2-3x more cheaply and needs no declared target_accept, because agents'
# variants of the Viterbi seed took ~40 min per fit. The seed keeps the name
# motif_stack: the harness withholds the seed whose name equals the held-out
# ground truth, and the ground-truth name scan looks for that name, so a seed
# called motif_stack_softmax would be handed to the agents when motif_stack is
# the hidden model and would trip the scan.
SEED_SOURCES = {"motif_stack": "motif_stack_softmax"}


def seed_source_name(name: str) -> str:
    """The registry model whose file the seed pool's ``name`` seed copies."""
    return SEED_SOURCES.get(name, name)


@dataclass
class Args:
    """Copy the registry's active model files + manifest into the seed pool."""

    check: bool = False
    """Only report drift (exit 1 if out of sync); copy nothing."""


def sync_seed_models(check: bool) -> int:
    names = read_manifest_names(REGISTRY_DIR)
    if not names:
        raise ValueError(f"Registry manifest lists no models: {REGISTRY_DIR}")

    # The two manifests each keep their own explanatory header, so they are not
    # byte-identical — but they must list the SAME models. A name mismatch is the
    # 2026-07 divergence, so refuse to sync bodies onto a mismatched name set.
    seed_names = read_manifest_names(SEED_POOL_DIR)
    if seed_names != names:
        raise ValueError(
            f"Seed manifest models {seed_names} != registry {names}; reconcile "
            f"{SEED_POOL_DIR / MANIFEST_FILENAME} before syncing bodies."
        )

    # Only the model bodies are a verbatim mirror (of SEED_SOURCES' file for
    # the one seed it names).
    drifted: list[str] = []
    for name in names:
        src = REGISTRY_DIR / f"{seed_source_name(name)}.py"
        dst = SEED_POOL_DIR / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Registry is missing {src.name}: {src}")
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            drifted.append(f"{name}.py")
            if not check:
                shutil.copyfile(src, dst)

    if check:
        if drifted:
            print(f"seed pool out of sync ({len(drifted)}): {', '.join(drifted)}")
            return 1
        print("seed pool is in sync with the registry")
        return 0

    if drifted:
        print(f"synced {len(drifted)} file(s) to the seed pool: {', '.join(drifted)}")
    else:
        print("seed pool already in sync; nothing to copy")
    return 0


def main(args: Args) -> None:
    sys.exit(sync_seed_models(check=args.check))


if __name__ == "__main__":
    main(tyro.cli(Args))
