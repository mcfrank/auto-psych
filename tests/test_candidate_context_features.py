"""The inner-loop CONTEXT.md must advertise the extensible-feature hook.

A candidate agent should learn that (a) only numeric columns can be a pm.Data,
(b) the raw sequence_a/sequence_b strings are not usable directly, and (c) it can
define compute_features(sequence_a, sequence_b) to derive new numeric features
the precomputed columns do not provide.
"""

from __future__ import annotations

import yaml

from src.pipelines.inner_loop.pymc_orchestrator import _write_candidate_context

HEADER = (
    "participant_id,trial_index,sequence_a,sequence_b,chose_left,chose_right,model,"
    "n_a,h_a,p_a,n_b,h_b,p_b"
)


def _setup(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "People do X."}]}),
        encoding="utf-8",
    )
    responses = tmp_path / "responses.csv"
    responses.write_text(HEADER + "\n1,0,HTH,HHT,1,0,seed,3,2,0.67,3,2,0.67\n")
    cand_dir = tmp_path / "iter_0" / "candidate_0"
    return responses, models_dir, cand_dir


def test_context_advertises_compute_features_hook(tmp_path):
    responses, models_dir, cand_dir = _setup(tmp_path)
    _write_candidate_context(
        cand_dir, responses, models_dir, 0, 0, 3, current_posterior=None
    )
    context = (cand_dir / "CONTEXT.md").read_text(encoding="utf-8")

    assert "compute_features" in context
    # Raw sequences are named as the source for derived features.
    assert "sequence_a" in context and "sequence_b" in context
    # The agent is told numeric-only is the rule for pm.Data inputs.
    assert "numeric" in context.lower()


def test_context_does_not_tell_agent_to_bind_raw_string_columns(tmp_path):
    """The old text listed every column (incl. strings) as a pm.Data name."""
    responses, models_dir, cand_dir = _setup(tmp_path)
    _write_candidate_context(
        cand_dir, responses, models_dir, 0, 0, 3, current_posterior=None
    )
    context = (cand_dir / "CONTEXT.md").read_text(encoding="utf-8")

    assert "use these as `pm.Data` names" not in context
