"""Unit tests for the LLM-as-judge model-similarity machinery.

The judge backend is injected, so every test drives a fake ``judge_fn`` — no
network, no API key. The fakes also record their calls so we can assert on
ordering (symmetrization) and caching behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.subjective_randomness.model_similarity_judge import (
    build_user_prompt,
    judge_pair,
    load_hypothesis,
    parse_rating,
    plot_similarity_trajectories,
    run_similarity,
    similarity_trajectory,
)


# ── parse_rating ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"rating": 5, "rationale": "both use alternation rate"}', 5),
        ("Rating: 3", 3),
        ("I'd say 6/7 — same prototype mechanism.", 6),
        ("The answer is 2.", 2),
        ('  {"rationale": "x", "rating": 7}  ', 7),
    ],
)
def test_parse_rating_extracts_score(reply, expected):
    assert parse_rating(reply) == expected


@pytest.mark.parametrize("reply", ["no number here", "rating: 9", "{}", "3 and 5 and 6"])
def test_parse_rating_fails_loudly_when_absent_or_ambiguous(reply):
    with pytest.raises(ValueError, match="parse"):
        parse_rating(reply)


# ── build_user_prompt ───────────────────────────────────────────────────────


def test_build_user_prompt_contains_both_sources_blind():
    prompt = build_user_prompt("AAA_code", "BBB_code")
    assert "AAA_code" in prompt and "BBB_code" in prompt
    assert "Model A" in prompt and "Model B" in prompt
    # The judge must not be told which is the ground truth.
    assert "ground truth" not in prompt.lower()


def test_build_user_prompt_includes_hypotheses_when_given():
    prompt = build_user_prompt(
        "AAA_code", "BBB_code", hypothesis_a="alpha mechanism", hypothesis_b="beta mechanism"
    )
    assert "alpha mechanism" in prompt and "beta mechanism" in prompt
    assert "Stated hypothesis" in prompt
    # Code is still present alongside the hypothesis.
    assert "AAA_code" in prompt and "BBB_code" in prompt


# ── load_hypothesis: file then manifest fallback ────────────────────────────


def test_load_hypothesis_prefers_file_then_manifest(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "models_manifest.yaml").write_text(
        "models:\n  - name: foo\n    rationale: from manifest\n"
        "  - name: bar\n    rationale: bar rationale\n"
    )
    (d / "foo.hypothesis.md").write_text("from file\n")
    assert load_hypothesis(d, "foo") == "from file"  # explicit file wins
    assert load_hypothesis(d, "bar") == "bar rationale"  # manifest fallback
    assert load_hypothesis(d, "missing") is None  # neither -> None


# ── judge_pair: symmetrization + caching ────────────────────────────────────


class _RecordingJudge:
    """Returns a fixed rating per ordered prompt; records each (system,user)."""

    def __init__(self, rating_for_user):
        self.rating_for_user = rating_for_user
        self.calls = []

    def __call__(self, system, user):
        self.calls.append(user)
        return json.dumps({"rating": self.rating_for_user(user), "rationale": "x"})


def test_judge_pair_symmetrizes_by_averaging_both_orders():
    # Rating depends on which code appears first -> position bias of 4 vs 6.
    judge = _RecordingJudge(lambda user: 6 if user.index("CODE_A") < user.index("CODE_B") else 4)
    out = judge_pair("CODE_A", "CODE_B", judge_fn=judge, symmetrize=True)
    assert out["ratings"] == [6, 4]
    assert out["similarity"] == pytest.approx(5.0)
    assert len(judge.calls) == 2  # both orders judged


def test_judge_pair_single_order_when_not_symmetrized():
    judge = _RecordingJudge(lambda user: 5)
    out = judge_pair("CODE_A", "CODE_B", judge_fn=judge, symmetrize=False)
    assert out["ratings"] == [5]
    assert len(judge.calls) == 1


def test_judge_pair_uses_cache_to_avoid_recalls():
    judge = _RecordingJudge(lambda user: 5)
    cache = {}
    judge_pair("CODE_A", "CODE_B", judge_fn=judge, cache=cache, symmetrize=True)
    n_first = len(judge.calls)
    judge_pair("CODE_A", "CODE_B", judge_fn=judge, cache=cache, symmetrize=True)
    assert len(judge.calls) == n_first  # second call fully served from cache


# ── trajectory walking over a tiny fixture run ──────────────────────────────


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    seed_dir = tmp_path / "seed_models"
    seed_dir.mkdir()
    (seed_dir / "gt.py").write_text("# ground truth mechanism\nGT = 1\n")

    run_root = tmp_path / "run"
    for exp in (1, 2):
        models = run_root / f"experiment{exp}" / "model_loop" / "models"
        models.mkdir(parents=True)
        (models / "seed_a.py").write_text("# seed a\nA = 1\n")
        (models / "cand.py").write_text("# candidate\nC = 1\n")
        history = [
            {"step": 0, "iteration": None, "best_model": "seed_a"},
            {"step": 1, "iteration": 0, "best_model": "cand"},
        ]
        (run_root / f"experiment{exp}" / "model_loop" / "history.json").write_text(
            json.dumps(history)
        )
    return run_root, seed_dir


def test_similarity_trajectory_walks_every_step(tmp_path):
    run_root, seed_dir = _write_run(tmp_path)
    judge = _RecordingJudge(lambda user: 4)
    rows = similarity_trajectory(
        run_root, "gt", seed_models_dir=seed_dir, n_experiments=2,
        judge_fn=judge, symmetrize=False,
    )
    assert [r["global_step"] for r in rows] == [0, 1, 2, 3]
    assert [r["best_model"] for r in rows] == ["seed_a", "cand", "seed_a", "cand"]
    assert all(r["similarity"] == 4 for r in rows)


def test_similarity_trajectory_fails_on_missing_best_model(tmp_path):
    run_root, seed_dir = _write_run(tmp_path)
    # Point a history entry at a nonexistent model file.
    hist_path = run_root / "experiment1" / "model_loop" / "history.json"
    hist = json.loads(hist_path.read_text())
    hist[0]["best_model"] = "ghost"
    hist_path.write_text(json.dumps(hist))
    with pytest.raises(FileNotFoundError, match="Best-model source missing"):
        similarity_trajectory(
            run_root, "gt", seed_models_dir=seed_dir, n_experiments=2,
            judge_fn=_RecordingJudge(lambda user: 4), symmetrize=False,
        )


def test_run_similarity_and_plot(tmp_path):
    run_root, seed_dir = _write_run(tmp_path)
    result = {
        "seed_models_dir": str(seed_dir),
        "n_experiments": 2,
        "gt_runs": [{"gt_model": "gt", "run_root": str(run_root)}],
    }
    sim = run_similarity(result, judge_fn=_RecordingJudge(lambda user: 5), symmetrize=False)
    assert sim["gt_runs"][0]["trajectory"][-1]["similarity"] == 5

    # Overlay a fake holdout correlation trajectory; figure must render.
    holdout = {
        "gt_runs": [
            {
                "gt_model": "gt",
                "trajectory": [
                    {"global_step": i, "pearson_r": 0.9} for i in range(4)
                ],
            }
        ]
    }
    out_path = tmp_path / "figs" / "sim.png"
    plot_similarity_trajectories(sim, out_path, holdout_result=holdout)
    assert out_path.exists() and out_path.stat().st_size > 0
