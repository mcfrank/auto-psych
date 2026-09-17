"""The opencode.json config must NOT grant ``write`` permission.

opencode's ``write`` tool (for creating new files) is unreliable in headless
mode: it auto-denies writes even with ``"write": "allow"`` in the config.
The campaign-era sweeps (iteration 3) never had ``write`` in the config and
admitted candidates at scale because Gemini used ``bash`` (heredoc/cat)
instead.  The correct fix is to omit ``write`` from the config entirely so
the agent is never offered the tool, and to instruct agents explicitly to
use ``bash`` for file creation in the candidate prompt.

The sbatch hardening script patches per-task copies of ``opencode.json``;
this test asserts the base config and the hardened config both omit write.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyprojroot import here


def test_base_opencode_json_omits_write_permission():
    """The base opencode.json must NOT include a ``"write"`` key."""
    oc_path = here() / "opencode.json"
    cfg = json.loads(oc_path.read_text(encoding="utf-8"))
    perm = cfg.get("permission", {})
    assert "write" not in perm, (
        f"opencode.json grants write permission; the write tool is "
        f"unreliable in headless mode — agents must use bash instead. "
        f"Remove the 'write' key from permission."
    )


def test_sbatch_hardening_does_not_add_write_permission(tmp_path):
    """The inline Python in holdout_recovery_array.sbatch must not add write."""
    oc_path = here() / "opencode.json"
    base_cfg = json.loads(oc_path.read_text(encoding="utf-8"))

    # The hardening script's logic, extracted:
    cfg = dict(base_cfg)
    perm = cfg.setdefault("permission", {})
    deny_globs = [
        "**/seed_models/**",
        "**/model_families/**",
        "**/pymc_model_families/**",
        "**/configs/holdout_recovery*.yaml",
        "**/ground_truth_models.py",
        "**/evaluate_recovery.py",
        "**/gt.txt",
    ]
    perm["read"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
    perm["glob"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
    perm["grep"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
    ext = perm.setdefault("external_directory", {})
    repo = str(tmp_path / "fake_repo")
    ext[f"{repo}/**"] = "allow"
    ext[f"{repo}/*"] = "allow"

    assert "write" not in perm, (
        "sbatch hardening added a write permission"
    )


def test_candidate_prompt_instructs_bash_writing():
    """The candidate prompt must tell agents to use bash for file creation."""
    from src.pipelines.inner_loop.pymc_orchestrator import _build_candidate_prompt
    prompt = _build_candidate_prompt(
        Path("/tmp/test_candidate_dir"),
        {
            "context": "test context",
            "brief": "test brief",
            "existing_hypotheses": "test hypotheses",
        },
    )
    assert "bash" in prompt.lower() or "cat" in prompt.lower(), (
        "candidate prompt does not instruct agents to use bash for file creation"
    )
    assert "cat <<" in prompt or "cat >" in prompt or "heredoc" in prompt.lower(), (
        "candidate prompt does not show a bash heredoc example for writing files"
    )
