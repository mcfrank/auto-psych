"""The candidate agent's context is injected into its prompt, not left on disk.

Leaving CONTEXT.md / CANDIDATE_BRIEF.md / existing_hypotheses.md / critiques.md
as files the agent must open itself made the round brief and the critique
optional reading — an agent that skipped them lost exactly the information
meant to steer exploration. The prompt now inlines all four documents as
delimited sections (the files are still written for audit/reproducibility).
"""

from __future__ import annotations

from src.pipelines.inner_loop.pymc_orchestrator import (
    _build_candidate_prompt,
    _write_candidate_context,
)


def test_prompt_inlines_all_context_documents(tmp_path):
    docs = {
        "context": "RESPONSES CSV COLUMNS: n_a,h_a",
        "brief": "Propose a mechanism the current models cannot express.",
        "existing_hypotheses": "- seed_a: mechanism seed_a (posterior 0.9)",
        "critiques": "## overalternation_rate — observed 0.7, model 0.5",
    }
    prompt = _build_candidate_prompt(tmp_path / "candidate_0", docs)
    for text in docs.values():
        assert text in prompt
    # The instruction still names the working dir and the three output files.
    assert "candidate_0" in prompt
    assert "hypothesis.md" in prompt
    assert "model_name.txt" in prompt
    assert "candidate.py" in prompt


def test_prompt_omits_critiques_section_when_absent(tmp_path):
    docs = {
        "context": "ctx",
        "brief": "brief",
        "existing_hypotheses": "hyps",
        "critiques": None,
    }
    prompt = _build_candidate_prompt(tmp_path / "candidate_0", docs)
    assert "critiques.md" not in prompt


def test_write_candidate_context_returns_the_written_documents(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: seed_a\n    rationale: mechanism seed_a\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.csv"
    responses.write_text("n_a,h_a,chose_left\n4,2,1\n", encoding="utf-8")
    candidate_dir = tmp_path / "iter_0" / "candidate_0"

    docs = _write_candidate_context(
        candidate_dir,
        responses,
        models_dir,
        iteration=0,
        candidate_idx=0,
        candidate_count=3,
        current_posterior=None,
    )

    assert docs["context"] == (candidate_dir / "CONTEXT.md").read_text(
        encoding="utf-8"
    )
    assert docs["brief"] == (candidate_dir / "CANDIDATE_BRIEF.md").read_text(
        encoding="utf-8"
    )
    assert docs["existing_hypotheses"] == (
        candidate_dir / "existing_hypotheses.md"
    ).read_text(encoding="utf-8")
    assert docs["critiques"] is None
