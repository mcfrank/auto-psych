"""The two time limits the user set (2026-09-28).

- A candidate's admission fit gets 30 minutes per sampling run (was 15);
  the rejection text, the repair note and the candidate brief all read the
  one constant.
- Live human data collection waits 3 hours (was 2) before it gives up and
  pauses the study; every path (run.py, the launchers' jobs, a resume)
  goes through the one constant in collect.py.
"""

from __future__ import annotations

import inspect

from src.models import mcmc_defaults
from src.pipelines.outer_loop import collect


def test_a_candidate_fit_gets_thirty_minutes_per_sampling_run():
    assert mcmc_defaults.CANDIDATE_FIT_TIME_LIMIT_SEC == 30 * 60


def test_live_collection_waits_three_hours():
    assert collect._PROLIFIC_MAX_WAIT_SEC == 3 * 60 * 60
    default = inspect.signature(collect._poll_prolific_until_target).parameters["max_wait_sec"]
    assert default.default == 3 * 60 * 60
