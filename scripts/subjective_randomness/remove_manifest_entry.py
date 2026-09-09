"""Remove one model from a ``models_manifest.yaml`` — the holdout sandbox scrub.

The holdout recovery array task (``slurm/holdout_recovery_array.sbatch``) gives
each task its own sanitised copy of the repo and deletes the held-out model's
``.py`` from both model directories the agents can open (the GT registry and
the live seed pool). The manifests beside those files still listed the held-out
model with its ``rationale`` — a one-sentence statement of the very mechanism
the loop is supposed to rediscover. This script removes that entry (and, by
re-serialising the file, the header comment that names every active model),
then verifies the name is gone.

    python scripts/subjective_randomness/remove_manifest_entry.py \\
        --models-dir <repo copy>/src/subjective_randomness/pymc_model_families \\
        --name motif_stack

An unlisted name is an error, unless ``--missing-ok`` says a superseded ground
truth (kept on disk for refits but not in the live manifests) is expected.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.models.model_manifest import (  # noqa: E402
    manifest_path,
    read_manifest_names,
    remove_manifest_entry,
)


@dataclass
class Args:
    """Remove one model's entry from a models_manifest.yaml."""

    models_dir: Path
    """Directory whose ``models_manifest.yaml`` is rewritten in place."""
    name: str
    """The model to remove (its ``.py`` should already be gone)."""
    missing_ok: bool = False
    """Accept a name the manifest does not list (a superseded ground truth);
    without this flag an unlisted name is an error."""


def main(args: Args) -> None:
    manifest = manifest_path(args.models_dir)
    removed = remove_manifest_entry(args.models_dir, args.name)
    if removed:
        print(f"[manifest] removed {args.name!r} from {manifest}", flush=True)
    elif args.missing_ok:
        print(
            f"[manifest] {args.name!r} is not listed in {manifest} — nothing to remove",
            flush=True,
        )
    else:
        raise SystemExit(
            f"{args.name!r} is not listed in {manifest}; pass --missing-ok if a "
            f"superseded (unlisted) ground truth is expected."
        )
    remaining = read_manifest_names(args.models_dir)
    if args.name in remaining:
        raise RuntimeError(
            f"{manifest} still lists {args.name!r} after the rewrite: {remaining}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
