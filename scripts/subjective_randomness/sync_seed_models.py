"""CLI: resync the outer loop's live seed pool from the recovery registry.

The registry at ``src/subjective_randomness/pymc_model_families/`` is the single
source of truth for the active seed set. The outer loop's live seed pool at
``src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`` is a
verbatim mirror (it is copied into experiment 1 and shown to coding agents as
worked examples). Edit the registry, then run this to propagate the change;
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

    # Only the model bodies are a verbatim mirror.
    drifted: list[str] = []
    for name in names:
        src = REGISTRY_DIR / f"{name}.py"
        dst = SEED_POOL_DIR / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Registry is missing {name}.py: {src}")
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
