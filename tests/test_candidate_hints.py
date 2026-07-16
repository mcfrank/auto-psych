"""The exploration hints steering each round's candidates are broad and configurable.

The old hard-coded 3-hint rotation capped per-round diversity: only one hint in
three pushed genuine novelty, and raising candidate_count just re-cycled the
same three hints. The default set is now a larger battery of distinct
theoretical lenses (one per candidate when candidate_count <= len(hints)), and
the whole list is a parameter so a hero config can tune it without editing
source.
"""

from __future__ import annotations

from src.pipelines.inner_loop.pymc_orchestrator import (
    DEFAULT_CANDIDATE_HINTS,
    _write_candidate_context,
)


def _context(tmp_path, idx, hints=None, count=None):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: seed_a\n    rationale: mechanism seed_a\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.csv"
    responses.write_text("n_a,chose_left\n4,1\n", encoding="utf-8")
    kwargs = {} if hints is None else {"hints": hints}
    return _write_candidate_context(
        tmp_path / f"candidate_{idx}",
        responses,
        models_dir,
        iteration=0,
        candidate_idx=idx,
        candidate_count=count or (len(hints) if hints else 7),
        current_posterior=None,
        **kwargs,
    )


def test_default_hint_set_is_a_broad_battery():
    assert len(DEFAULT_CANDIDATE_HINTS) >= 7
    assert len(set(DEFAULT_CANDIDATE_HINTS)) == len(DEFAULT_CANDIDATE_HINTS)
    assert all(hint.strip() for hint in DEFAULT_CANDIDATE_HINTS)


def test_each_candidate_gets_a_distinct_default_hint(tmp_path):
    briefs = [
        _context(tmp_path, idx)["brief"]
        for idx in range(len(DEFAULT_CANDIDATE_HINTS))
    ]
    for hint, brief in zip(DEFAULT_CANDIDATE_HINTS, briefs):
        assert hint in brief


def test_custom_hints_replace_the_default_set(tmp_path):
    hints = ["Use only the single custom lens for this run."]
    docs = _context(tmp_path, 0, hints=hints, count=3)
    assert hints[0] in docs["brief"]
    assert DEFAULT_CANDIDATE_HINTS[0] not in docs["brief"]
