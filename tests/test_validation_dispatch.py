"""``run_validation`` must not report success for a stage it cannot validate.

The dispatcher used to return ``validation_ok: True`` for any ``agent_key``
missing from ``AGENT_VALIDATORS``. A renamed or mistyped stage therefore read
as "validated and fine", which is the most expensive possible way to be wrong:
the pipeline advances on unchecked output.
"""

from __future__ import annotations

import pytest

from src.validation.validators import AGENT_VALIDATORS, run_validation


def test_unknown_stage_raises_and_names_the_known_stages(tmp_path):
    state = {"project_id": "subjective_randomness", "run_id": 1, "run_dir": str(tmp_path)}
    with pytest.raises(KeyError) as excinfo:
        run_validation(state, "7_publish")
    message = str(excinfo.value)
    assert "7_publish" in message
    assert "1_theory" in message  # the caller is told what IS dispatchable


def test_every_known_stage_is_dispatchable():
    # No "2_design": the design stage is programmatic (exhaustive EIG), not an
    # agent, so it has no validator to dispatch.
    assert set(AGENT_VALIDATORS) == {
        "1_theory",
        "3_implement",
        "4_collect",
        "5_analyze",
        "6_interpret",
    }
