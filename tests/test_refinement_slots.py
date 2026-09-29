"""Units of the depth mechanism: slot roles, the lens walk over exploratory
slots, the refinement menu and the per-slot briefs.

``tests/test_refinement_slots_loop.py`` drives the whole loop; these tests
pin the pieces it is built from.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src.pipelines.inner_loop.candidate_agent import (
    DEFAULT_CANDIDATE_HINTS,
    _build_candidate_prompt,
    _write_candidate_context,
    _write_refinement_menu,
)
from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
    LedgerEntry,
)
from src.pipelines.inner_loop.model_zoo import (
    SLOT_EXPLORE,
    SLOT_REFINE_CHOSEN,
    SLOT_REFINE_INCUMBENT,
    _lens_index,
    _lens_offset,
    exploratory_slots_per_round,
    parse_prune_margin,
    prune_margin_detail,
    slot_roles,
)

E, I, C = SLOT_EXPLORE, SLOT_REFINE_INCUMBENT, SLOT_REFINE_CHOSEN


# ── Allocation ─────────────────────────────────────────────────────────


def test_slot_roles_allocation_at_every_count():
    """At four or more slots: C - 3 exploratory, two incumbent, one chosen.
    Below four the refinement slots go one at a time — the second incumbent
    slot first, then the chosen slot, then the last incumbent slot — and the
    exploratory slot never goes."""
    assert slot_roles(1) == [E]
    assert slot_roles(2) == [E, I]
    assert slot_roles(3) == [E, I, C]
    assert slot_roles(4) == [E, I, I, C]
    assert slot_roles(5) == [E, E, I, I, C]
    assert slot_roles(6) == [E, E, E, I, I, C]
    assert slot_roles(9) == [E] * 6 + [I, I, C]
    # A round with no slots (max_iterations rounds that only score) is legal.
    assert slot_roles(0) == []
    with pytest.raises(ValueError, match="negative"):
        slot_roles(-1)


def test_exploratory_slots_per_round():
    assert [exploratory_slots_per_round(c) for c in range(7)] == [0, 1, 1, 1, 1, 2, 3]


# ── The lens walk is over exploratory slots only ───────────────────────


def test_lens_offset_counts_exploratory_slots_only():
    """Experiment k spends ``max_iterations × exploratory slots`` lenses, so
    at three candidates (one exploratory slot) two rounds use two lenses."""
    assert _lens_offset(1, max_iterations=2, candidate_count=3) == 0
    assert _lens_offset(2, max_iterations=2, candidate_count=3) == 2
    assert _lens_offset(3, max_iterations=2, candidate_count=6) == 12
    assert _lens_offset(2, max_iterations=5, candidate_count=6) == 15


def test_six_candidate_rounds_walk_the_battery_without_a_repeat():
    """Three exploratory slots per round: no lens repeats within a round, and
    two experiments of two rounds fire every lens of the battery once."""
    n = len(DEFAULT_CANDIDATE_HINTS)
    fired = Counter()
    for exp_num in (1, 2):
        offset = _lens_offset(exp_num, max_iterations=2, candidate_count=6)
        for iteration in range(2):
            lenses = [_lens_index(offset, iteration, 3, i, n) for i in range(3)]
            assert len(set(lenses)) == 3
            fired.update(lenses)
    assert fired == Counter({lens: 1 for lens in range(n)})


# ── The prune margin: written by pruning, read back to rank the menu ──


def test_prune_margin_round_trips():
    detail = prune_margin_detail(elpd_diff=30.04, dse=5.0, baseline="model_a")
    assert detail == "30.0 nats behind model_a (6.0× dse)"
    assert parse_prune_margin(detail) == 30.0
    assert parse_prune_margin("415.5 nats behind local_representativeness (15.7× dse)") == 415.5
    with pytest.raises(ValueError, match="prune margin"):
        parse_prune_margin("no candidate.py written")


# ── Ledger: the pruned models, by name, not in the live set ────────────


def _entry(name, outcome, *, detail="", hypothesis="H.", context="experiment1 round 0"):
    return LedgerEntry(
        name=name, outcome=outcome, detail=detail, hypothesis=hypothesis, context=context
    )


def test_ledger_pruned_is_the_latest_pruned_outcome_per_name_not_live(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    ledger.append(_entry("lost", "admitted"))
    ledger.append(_entry("lost", "pruned", detail="30.0 nats behind seed (6.0× dse)"))
    ledger.append(_entry("never_entered", "rejected", detail="no candidate.py written"))
    ledger.append(_entry("back", "pruned", detail="9.0 nats behind seed (2.5× dse)"))
    ledger.append(_entry("back", "admitted"))  # re-admitted later: live now
    ledger.append(_entry("gone", "dropped", detail="cannot be fit"))

    pruned = ledger.pruned(live_names={"seed", "back"})

    assert [(e.name, e.detail) for e in pruned] == [
        ("lost", "30.0 nats behind seed (6.0× dse)")
    ]


# ── The refinement menu ────────────────────────────────────────────────


def _models_dir(tmp_path, entries):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    lines = ["models:"]
    for name, rationale in entries:
        (models_dir / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
        lines += [f"  - name: {name}", f"    rationale: {json.dumps(rationale)}"]
    (models_dir / "models_manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return models_dir


def _row(rank, diff, dse):
    return {
        "rank": rank,
        "elpd_loo": -10.0 - diff,
        "elpd_diff": diff,
        "dse": dse,
        "weight": 0.0,
        "loo_unreliable": False,
    }


LONG = " ".join(f"Sentence {i} of a long hypothesis about runs." for i in range(60))


def _menu_fixture(tmp_path):
    models_dir = _models_dir(
        tmp_path,
        [
            ("model_a", "A: the incumbent."),
            ("model_b", "B: the distant live rival."),
            ("model_c", "C: the close live rival."),
        ],
    )
    comparison = {
        "model_a": _row(0, 0.0, 0.0),
        "model_c": _row(1, 1.0, 5.0),
        "model_b": _row(2, 3.0, 5.0),
    }
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    ledger.append(_entry("far_idea", "admitted", hypothesis="Far: " + LONG))
    ledger.append(
        _entry(
            "far_idea",
            "pruned",
            detail="400.0 nats behind model_a (15.7× dse)",
            hypothesis="Far: " + LONG,
            context="experiment2 round 0",
        )
    )
    (models_dir / "pruned").mkdir(exist_ok=True)
    (models_dir / "pruned" / "far_idea.py").write_text("# far_idea\n", encoding="utf-8")
    # Pruned in an earlier experiment: in the inherited ledger, no file here.
    ledger.append(
        _entry(
            "near_idea",
            "pruned",
            detail="8.0 nats behind model_a (2.1× dse)",
            hypothesis="Near: people weigh the longest run.",
            context="experiment1 round 1",
        )
    )
    ledger.append(
        _entry("dup_idea", "rejected", detail="predicts like existing model 'model_a'")
    )
    return models_dir, comparison, ledger


def test_refinement_menu_ranks_live_by_standing_and_pruned_by_margin(tmp_path):
    models_dir, comparison, ledger = _menu_fixture(tmp_path)
    candidate_dir = tmp_path / "candidate_2"
    candidate_dir.mkdir()

    text = _write_refinement_menu(
        candidate_dir, models_dir, comparison, ledger, incumbent="model_a"
    )

    assert (candidate_dir / "refinement_menu.md").read_text(encoding="utf-8") == text
    live = text.split("## Live", 1)[1].split("## Pruned", 1)[0]
    pruned = text.split("## Pruned", 1)[1]
    # Live: the close rival before the distant one; the incumbent absent.
    assert live.index("### model_c") < live.index("### model_b")
    assert "model_a" not in live
    assert "C: the close live rival." in live
    assert str(models_dir / "model_c.py") in live
    assert "statistically tied with the best" in live
    # Pruned: narrowest margin first, full hypothesis, margin, source.
    assert pruned.index("### near_idea") < pruned.index("### far_idea")
    assert "8.0 nats behind model_a (2.1× dse)" in pruned
    assert "400.0 nats behind model_a (15.7× dse)" in pruned
    assert "Far: " + LONG in pruned
    assert "…" not in text
    assert str(models_dir / "pruned" / "far_idea.py") in pruned
    # A model pruned in an earlier experiment has no source in this tree; say so.
    assert "near_idea.py" not in pruned
    assert "earlier experiment" in pruned
    # Rejected candidates never entered the set: not on the menu.
    assert "dup_idea" not in text
    # A menu, not a blacklist.
    assert "do not re-propose" not in text.lower()
    assert "menu" in text.lower()


def test_refinement_menu_says_when_nothing_has_been_pruned(tmp_path):
    models_dir = _models_dir(tmp_path, [("model_a", "A."), ("model_b", "B.")])
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    candidate_dir = tmp_path / "candidate_2"
    candidate_dir.mkdir()
    text = _write_refinement_menu(candidate_dir, models_dir, {}, ledger, incumbent="model_a")
    assert "### model_b" in text
    assert "No model has been pruned" in text


def test_refinement_menu_requires_the_incumbent_to_be_live(tmp_path):
    models_dir = _models_dir(tmp_path, [("model_a", "A.")])
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)
    with pytest.raises(ValueError, match="incumbent"):
        _write_refinement_menu(tmp_path, models_dir, {}, ledger, incumbent="model_zzz")


# ── The per-slot briefs ────────────────────────────────────────────────


def _context(tmp_path, role, *, incumbent="model_a", ledger="default", comparison="default"):
    models_dir, default_comparison, default_ledger = _menu_fixture(tmp_path)
    responses = tmp_path / "responses.csv"
    responses.write_text("sequence_a,sequence_b,chose_left\nHHT,HTH,1\n", encoding="utf-8")
    candidate_dir = tmp_path / "iter_0" / "candidate_1"
    docs = _write_candidate_context(
        candidate_dir,
        responses,
        models_dir,
        iteration=0,
        candidate_idx=1,
        candidate_count=3,
        current_posterior=None,
        ledger=default_ledger if ledger == "default" else ledger,
        comparison=default_comparison if comparison == "default" else comparison,
        lens_index=4,
        role=role,
        incumbent=incumbent,
    )
    return candidate_dir, models_dir, docs


MEGA_MODEL_CLAUSE = "a blended mega-model is not a hypothesis"


def test_exploratory_slot_keeps_lens_blacklist_and_composition_rule(tmp_path):
    candidate_dir, _, docs = _context(tmp_path, SLOT_EXPLORE)
    assert DEFAULT_CANDIDATE_HINTS[4] in docs["brief"]
    assert MEGA_MODEL_CLAUSE in docs["brief"]
    assert "do not re-propose" in docs["attempted"].lower()
    assert docs["menu"] is None
    assert (candidate_dir / "attempted_hypotheses.md").exists()
    assert not (candidate_dir / "refinement_menu.md").exists()
    assert "Do not re-propose" in docs["context"]
    assert "refinement_menu.md" not in docs["context"]


def test_incumbent_slot_names_the_incumbent_and_lifts_the_composition_rules(tmp_path):
    candidate_dir, models_dir, docs = _context(tmp_path, SLOT_REFINE_INCUMBENT)
    brief = docs["brief"]
    assert "Refine the incumbent" in brief
    assert "`model_a`" in brief
    assert "A: the incumbent." in brief
    assert str(models_dir / "model_a.py") in brief
    assert "rank 0, the best model on this data" in brief
    assert MEGA_MODEL_CLAUSE not in brief
    assert "lifted" in brief
    assert not any(hint in brief for hint in DEFAULT_CANDIDATE_HINTS)
    assert "hypothesis.md" in brief  # say which model you refined, in prose
    # The retired list is a menu here, not a blacklist.
    assert docs["attempted"] is None
    assert docs["menu"] == (candidate_dir / "refinement_menu.md").read_text(encoding="utf-8")
    assert not (candidate_dir / "attempted_hypotheses.md").exists()
    assert "refinement_menu.md" in docs["context"]
    assert "Do not re-propose" not in docs["context"]


def test_chosen_slot_points_at_the_menu_and_excludes_the_incumbent(tmp_path):
    candidate_dir, _, docs = _context(tmp_path, SLOT_REFINE_CHOSEN)
    brief = docs["brief"]
    assert "of your choosing" in brief
    assert "`model_a`" in brief  # named as the one model this slot may not pick
    assert "refinement_menu.md" in brief
    assert MEGA_MODEL_CLAUSE not in brief
    assert "lifted" in brief
    assert not any(hint in brief for hint in DEFAULT_CANDIDATE_HINTS)
    assert docs["attempted"] is None
    assert "### near_idea" in docs["menu"]


def test_refinement_slot_requires_an_incumbent_and_a_ledger(tmp_path):
    with pytest.raises(ValueError, match="incumbent"):
        _context(tmp_path, SLOT_REFINE_INCUMBENT, incumbent=None)
    with pytest.raises(ValueError, match="ledger"):
        _context(tmp_path, SLOT_REFINE_CHOSEN, ledger=None)
    with pytest.raises(ValueError, match="role"):
        _context(tmp_path, "refine everything")


def test_prompt_inlines_the_menu_for_refinement_slots(tmp_path):
    docs = {
        "context": "ctx",
        "brief": "brief",
        "existing_hypotheses": "hyps",
        "attempted": None,
        "menu": "# Refinement menu\n\n### near_idea",
        "critiques": None,
        "attempt_note": None,
    }
    prompt = _build_candidate_prompt(tmp_path / "candidate_2", docs)
    assert "## refinement_menu.md\n\n# Refinement menu" in prompt
    assert "## attempted_hypotheses.md" not in prompt
    docs["menu"] = None
    assert "## refinement_menu.md" not in _build_candidate_prompt(tmp_path / "candidate_0", docs)
