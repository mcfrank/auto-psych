"""Going live on Prolific requires an explicit confirmation flag.

``prolific_mode: live`` recruits and PAYS real participants, yet it used to be
just another YAML value — the smoke-deploy script required
``--confirm-production`` while the actual money-spending path had no gate.
``--confirm-live-recruitment`` makes going live a second, conscious act; the
guard fires before any experiment work (or Prolific API call) starts.
"""

from __future__ import annotations

import pytest

from src.pipelines.outer_loop.run import Args, main


def test_live_without_confirmation_exits_before_any_work(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            Args(
                project="subjective_randomness",
                experiment=1,
                prolific_mode="live",
            )
        )
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "confirm-live-recruitment" in err


def test_live_with_confirmation_passes_the_gate(monkeypatch, capsys):
    # With the flag set, the gate must NOT fire — main proceeds to the next
    # validation (a nonexistent project), proving the exit above was the gate.
    with pytest.raises(SystemExit):
        main(
            Args(
                project="no_such_project",
                experiment=1,
                prolific_mode="live",
                confirm_live_recruitment=True,
            )
        )
    err = capsys.readouterr().err
    assert "confirm-live-recruitment" not in err


def test_non_live_modes_need_no_confirmation(capsys):
    # prolific_mode=none on a nonexistent project: fails on the project check,
    # not the gate.
    with pytest.raises(SystemExit):
        main(Args(project="no_such_project", experiment=1))
    err = capsys.readouterr().err
    assert "confirm-live-recruitment" not in err
