"""Check a candidate model the way admission will — before the agent finishes.

The candidate agent writes a PyMC model it has never run. This is the one
command ``CONTEXT.md`` documents for it (``check_candidate_command``): the
pipeline's own interpreter runs the admission gates the agent can act on —
the import allowlist, a loadable module-level ``model: pm.Model``, a finite
log-probability on the real responses, the data contract (observed data =
``chose_left`` in row order, a per-trial ``p_left`` that is the likelihood's
probability; ``src/models/model_contract.py``), a short MCMC fit and a finite
ELPD-LOO — with the small ``CANDIDATE_CHECK_*`` sampler settings from
``src/models/mcmc_defaults.py``. It is a smoke fit, not a production fit: it
never writes to the pipeline's fit cache, and it does not run the novelty
gate (that needs the other models' fits).

Run from the repository checkout (the agent's working directory)::

    <interpreter> -m src.pipelines.inner_loop.check_candidate \\
        --candidate-dir <dir holding candidate.py> --responses <responses.csv>

Prints an ``OK`` report and exits 0, or prints the exact reason admission
would reject the file and exits 1.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tyro

from src.model_comparison.likelihood import log_likelihood
from src.models.mcmc_defaults import (
    CANDIDATE_CHECK_CHAINS,
    CANDIDATE_CHECK_DRAWS,
    CANDIDATE_CHECK_TUNE,
)
from src.models.model_contract import model_contract_violation
from src.models.model_loading import load_pymc_model
from src.models.pymc_inference import fit_model, model_logp_is_finite
from src.pipelines.inner_loop.import_gate import (
    CANDIDATE_IMPORT_ALLOWLIST,
    check_forbidden_imports,
)

# The candidate file is always candidate.py, so its model name is fixed.
CANDIDATE_MODULE = "candidate"

# One chain on one core: cheap inside an agent session, and no worker
# processes to spawn. cache_dir stays None so a check never seeds the
# pipeline's on-disk fit cache with a smoke fit.
CHECK_FIT_KWARGS: Dict[str, Any] = {
    "draws": CANDIDATE_CHECK_DRAWS,
    "tune": CANDIDATE_CHECK_TUNE,
    "chains": CANDIDATE_CHECK_CHAINS,
    "cores": 1,
}


class CandidateCheckFailed(RuntimeError):
    """The candidate would be rejected at admission, for the stated reason."""


def check_candidate_command(candidate_dir: Path, responses_path: Path) -> str:
    """The exact shell command CONTEXT.md documents for this candidate.

    Uses this process's interpreter — the pipeline's own — so the agent runs
    the check with the PyMC stack the admission gates use, not whatever
    ``python3`` is on its PATH.
    """
    return (
        f"{sys.executable} -m src.pipelines.inner_loop.check_candidate \\\n"
        f"    --candidate-dir {candidate_dir} \\\n"
        f"    --responses {responses_path}"
    )


def run_candidate_check(candidate_dir: Path, responses_path: Path) -> str:
    """Run the admission gates on ``candidate_dir/candidate.py``; return the OK report.

    Raises ``CandidateCheckFailed`` with the same wording admission would use
    (see ``model_zoo._admit_candidate_with_reason``) at the first gate the
    candidate fails.
    """
    candidate_dir = Path(candidate_dir)
    responses_path = Path(responses_path)
    candidate_file = candidate_dir / f"{CANDIDATE_MODULE}.py"
    if not candidate_file.exists():
        raise CandidateCheckFailed(f"no candidate.py at {candidate_file}")
    if not responses_path.exists():
        raise CandidateCheckFailed(f"responses CSV not found: {responses_path}")

    forbidden = check_forbidden_imports(candidate_file.read_text(encoding="utf-8"))
    if forbidden:
        raise CandidateCheckFailed(
            f"forbidden import: {', '.join(forbidden)} — candidates may only "
            f"import from {sorted(CANDIDATE_IMPORT_ALLOWLIST)}"
        )

    try:
        load_pymc_model(CANDIDATE_MODULE, candidate_dir)
    except Exception as e:  # noqa: BLE001 — any load failure is the reason
        raise CandidateCheckFailed(
            f"candidate.py is not a loadable PyMC model: {e}"
        ) from e

    fittable, reason = model_logp_is_finite(
        CANDIDATE_MODULE, candidate_dir, responses_path
    )
    if not fittable:
        raise CandidateCheckFailed(f"model cannot be fit — {reason}")

    violation = model_contract_violation(
        CANDIDATE_MODULE, candidate_dir, responses_path
    )
    if violation is not None:
        raise CandidateCheckFailed(f"model breaks the data contract — {violation}")

    try:
        fit_model(
            CANDIDATE_MODULE,
            candidate_dir,
            responses_path,
            cache_dir=None,
            **CHECK_FIT_KWARGS,
        )
    except Exception as e:  # noqa: BLE001 — any sampling failure is the reason
        raise CandidateCheckFailed(
            f"MCMC sampling failed ({type(e).__name__}: {e})"
        ) from e

    # Same sampler settings ⇒ same in-process cache key ⇒ no second fit.
    try:
        elpd = log_likelihood(
            CANDIDATE_MODULE,
            responses_path,
            candidate_dir,
            cache_dir=None,
            **CHECK_FIT_KWARGS,
        )
    except Exception as e:  # noqa: BLE001 — any LOO failure is the reason
        raise CandidateCheckFailed(
            f"ELPD-LOO computation failed ({type(e).__name__}: {e})"
        ) from e
    if not math.isfinite(elpd):
        raise CandidateCheckFailed(
            f"non-finite ELPD-LOO ({elpd}); a model that assigns ~0 probability "
            "to an observed outcome would corrupt the posterior"
        )

    return (
        f"OK: {candidate_file} loads as a module-level pm.Model, has a finite "
        f"log-probability on {responses_path}, honours the data contract "
        f"(observed = chose_left, p_left = the likelihood's probability), completed a "
        f"{CANDIDATE_CHECK_DRAWS}-draw / {CANDIDATE_CHECK_TUNE}-tune smoke fit "
        f"({CANDIDATE_CHECK_CHAINS} chain) and has a finite ELPD-LOO ({elpd:.1f} "
        f"on this smoke fit). Not checked here: novelty against the other models."
    )


@dataclass
class Args:
    candidate_dir: Path
    """The candidate's working directory — the one holding candidate.py."""

    responses: Path
    """The responses CSV named in CONTEXT.md."""


def main(args: Args) -> None:
    try:
        report = run_candidate_check(args.candidate_dir, args.responses)
    except CandidateCheckFailed as e:
        print(
            f"CHECK FAILED — admission would reject this candidate:\n  {e}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from e
    print(report, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
