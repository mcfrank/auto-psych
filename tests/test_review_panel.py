"""A review panel runs members in file-based rounds: round 1 independent
notes, later rounds responding to the thread, then a moderator's plan.
Stages refuse to run on missing inputs, skip members whose note exists,
pause on a session limit, and validate every deliverable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from src.recovery_improvement.panel import (
    Panel,
    missing_inputs,
    parse_members,
    run_round,
    run_synthesis,
    verbal_digest,
)
from src.recovery_improvement.session_limit import SessionLimitHit

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
MEMBER_TEMPLATE = "round $round member $member lens $lens_title\nTHREAD:\n$thread\nVERBAL:\n$digest_verbal\nAUTO:\n$digest_autopsych\n$repair_feedback"
SYNTH_TEMPLATE = "moderator $moderator\n$members_list\nTHREAD:\n$thread\nDEFAULTS:\n$sweep_defaults\n$repair_feedback"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.fixture
def panel(tmp_path):
    source = tmp_path / "src_repo"
    (source / "scripts/subjective_randomness/configs").mkdir(parents=True)
    (source / "scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml").write_text("gt_models:\n  a:\n", encoding="utf-8")
    _git(source, "init", "-q"); _git(source, "add", "-A"); _git(source, "commit", "-q", "-m", "seed")
    verbal = tmp_path / "verbal"
    study = verbal / "data/cog-models/agentic-model-recovery/study-x/summary"
    study.mkdir(parents=True)
    (study / "recovery.csv").write_text(
        "model,replicate,round,incumbent_entry_id,kl_forward_nats,kl_backward_nats,kl_symmetric_nats\n"
        "best_first,0,-1,seed,1.0,1.0,2.0\nbest_first,0,3,cand,0.1,0.1,0.2\n"
        "best_first,1,-1,seed,1.0,1.0,3.0\nbest_first,1,3,cand,0.1,0.1,0.4\n", encoding="utf-8")
    root = tmp_path / "panel"
    root.mkdir()
    (root / "panel.env").write_text(
        f"PANEL_NAME=demo\nSOURCE_REPO={source}\nVERBAL_REPO={verbal}\n"
        "MEMBERS=\"m1:claude:claude-x:methods m2:codex:gpt-x:search\"\nMODERATOR=m1\n"
        "N_DISCUSSION_ROUNDS=1\nMEMBER_MAX_TURNS=1\nMEMBER_TIMEOUT_SEC=1\nMEMBER_MAX_BUDGET_USD=1\n"
        "PROMPT_ONLY_BACKENDS=\n",  # both fakes write files; the prompt-only test opts in
        encoding="utf-8")
    return Panel.load(root)


def test_parse_members_validates_every_field():
    members = parse_members("a:claude:m:methods b:codex:n:search")
    assert [m.name for m in members] == ["a", "b"] and members[1].backend == "codex"
    for bad in ("a:claude:m", "a:gemini:m:methods", "a:claude:m:nolens", "a:claude:m:methods a:codex:n:search"):
        with pytest.raises(ValueError):
            parse_members(bad)


def test_verbal_digest_reports_seed_vs_final_kl(panel):
    digest = verbal_digest(panel.verbal_repo)
    assert "| study-x | best_first | 2 | kl_symmetric_nats | 2.50 | n/a | 0.30 | 3 |" in digest


def test_verbal_digest_reads_the_older_column_layout_and_flags_unknown_ones(panel):
    old = panel.verbal_repo / "data/cog-models/agentic-model-recovery/study-old/summary"
    old.mkdir(parents=True)
    (old / "recovery.csv").write_text(
        "model,replicate,round,behavioral_distance,kl_per_action_nats,incumbent_entry_id\n"
        "bfs,0,0,0.7,0.80,seed__dfs\nbfs,0,5,0.1,0.10,iter_5__x\n", encoding="utf-8")
    odd = panel.verbal_repo / "data/cog-models/agentic-model-recovery/study-odd/summary"
    odd.mkdir(parents=True)
    (odd / "recovery.csv").write_text("model,score\nbfs,1\n", encoding="utf-8")
    digest = verbal_digest(panel.verbal_repo)
    assert "| study-old | bfs | 1 | kl_per_action_nats | n/a | 0.80 | 0.10 | 5 |" in digest
    assert "study-odd | (recovery.csv has no KL column I know; columns: model, score)" in digest


def _writer(text_by_label):
    """A fake agent that writes whatever note the brief names, recording briefs."""
    briefs = []

    def agent_for(member, label):
        def run(prompt, *, cwd, log_path):
            briefs.append((member.name, label, prompt))
            note = Path(prompt.split("NOTE=")[1].splitlines()[0]) if "NOTE=" in prompt else None
            return True, text_by_label.get(label, "done")
        return run
    return agent_for, briefs


def _note_writing_agent(panel, briefs, *, limit_for=None):
    def agent_for(member, label):
        def run(prompt, *, cwd, log_path):
            briefs.append((member.name, label, prompt))
            if limit_for == member.name:
                return True, "You've hit your session limit · resets 12am (America/Los_Angeles)"
            stage = "synthesis" if label == "synthesis" else int(label.rsplit("round", 1)[1])
            if stage == "synthesis":
                panel.synthesis_dir.mkdir(exist_ok=True)
                panel.plan_path.write_text("# plan\n" + "consensus " * 60, encoding="utf-8")
                (panel.synthesis_dir / "next_run.env").write_text("NOTE=test item 1\n", encoding="utf-8")
            else:
                panel.note_path(stage, member).write_text(f"# {member.name} round {stage}\n" + "finding " * 60, encoding="utf-8")
            return True, "done"
        return run
    return agent_for


def test_rounds_then_synthesis_build_a_thread_and_a_plan(panel):
    briefs = []
    agent_for = _note_writing_agent(panel, briefs)
    assert missing_inputs(panel, "2")  # round 2 needs round 1
    ran = run_round(panel, 1, agent_for=agent_for, member_template=MEMBER_TEMPLATE)
    assert ran == ["m1", "m2"]
    assert "independent round" in briefs[0][2] and "study-x" in briefs[0][2]
    assert not missing_inputs(panel, "2")
    run_round(panel, 2, agent_for=agent_for, member_template=MEMBER_TEMPLATE)
    round2_brief = briefs[2][2]
    assert "m2 round 1" in round2_brief and "Round 1 — m2" in round2_brief  # sees the other's note
    plan = run_synthesis(panel, agent_for=agent_for, synthesis_template=SYNTH_TEMPLATE)
    assert plan.is_file() and briefs[-1][1] == "synthesis" and briefs[-1][0] == "m1"
    assert "Round 2 — m1" in briefs[-1][2]
    thread = (panel.root / "thread.md").read_text(encoding="utf-8")
    assert thread.count("### Round") == 4


def test_members_with_a_note_are_skipped_and_a_limit_pauses_the_round(panel):
    briefs = []
    panel.round_dir(1).mkdir()
    panel.note_path(1, panel.members[0]).write_text("already here " * 30, encoding="utf-8")
    with pytest.raises(SessionLimitHit):
        run_round(panel, 1, agent_for=_note_writing_agent(panel, briefs, limit_for="m2"), member_template=MEMBER_TEMPLATE)
    assert [b[0] for b in briefs] == ["m2"]  # m1 skipped, m2 paused (no repair round)
    ran = run_round(panel, 1, agent_for=_note_writing_agent(panel, briefs), member_template=MEMBER_TEMPLATE)
    assert ran == ["m2"]


def test_missing_note_gets_one_repair_then_fails(panel):
    briefs = []

    def lazy(member, label):
        def run(prompt, *, cwd, log_path):
            briefs.append(prompt)
            return True, "forgot"
        return run
    with pytest.raises(RuntimeError, match="still invalid"):
        run_round(panel, 1, agent_for=lazy, member_template=MEMBER_TEMPLATE)
    assert len(briefs) == 2 and "REPAIR REQUIRED" in briefs[1]


def test_prompt_only_member_gets_the_evidence_pack_and_delivers_by_message(panel, tmp_path):
    evidence = tmp_path / "orchestrator.py"
    evidence.write_text("def admit(): return 'novelty gate'\n", encoding="utf-8")
    with (panel.root / "panel.env").open("a", encoding="utf-8") as fh:
        fh.write(f"EVIDENCE_FILES={evidence} {tmp_path / 'missing.md'}\nMODERATOR=m2\nPROMPT_ONLY_BACKENDS=codex\n")
    panel = Panel.load(panel.root)
    assert panel.is_prompt_only(panel.members[1]) and not panel.is_prompt_only(panel.members[0])
    briefs = []

    def agent_for(member, label):
        def run(prompt, *, cwd, log_path):
            briefs.append((member.name, prompt))
            if member.name == "m1":
                panel.note_path(1, member).write_text("m1 findings " * 30, encoding="utf-8")
                return True, "done"
            if label == "synthesis":
                return True, "# plan\n" + "consensus " * 40 + "\n```next_run.env\nNOTE=test item 1\nDRAWS=1000\n```\n"
            return True, "# m2 note\n" + "reasoned from the pack " * 20
        return run

    run_round(panel, 1, agent_for=agent_for, member_template=MEMBER_TEMPLATE + "\nCAP: $capabilities\nEV: $evidence")
    m1_brief = next(p for n, p in briefs if n == "m1")
    m2_brief = next(p for n, p in briefs if n == "m2")
    assert "novelty gate" in m2_brief and "(missing on disk)" in m2_brief and "cannot run commands" in m2_brief
    assert "novelty gate" not in m1_brief and "not inlined" in m1_brief
    assert panel.note_path(1, panel.members[1]).read_text(encoding="utf-8").startswith("# m2 note")
    # A prompt-only moderator: the plan is its message, the sweep spec its fenced block.
    (panel.root / "round2").mkdir()
    for m in panel.members:
        panel.note_path(2, m).write_text("round 2 " * 40, encoding="utf-8")
    run_synthesis(panel, agent_for=agent_for, synthesis_template=SYNTH_TEMPLATE + "\n$capabilities")
    assert panel.plan_path.read_text(encoding="utf-8").startswith("# plan")
    assert (panel.synthesis_dir / "next_run.env").read_text(encoding="utf-8") == "NOTE=test item 1\nDRAWS=1000\n"
