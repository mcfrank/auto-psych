"""The opencode.json config must grant ``write`` permission.

opencode treats ``edit`` (modify existing files) and ``write`` (create new
files) as separate tools with separate permission keys.  Without
``"write": "allow"``, every candidate agent that calls the write tool to
create ``candidate.py`` / ``hypothesis.md`` gets auto-rejected — no model
ever enters the zoo.

The sbatch hardening script patches per-task copies of ``opencode.json``;
this test asserts that the BASE config (which it starts from) already
includes write.  A separate test asserts the hardening script preserves it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyprojroot import here


def test_base_opencode_json_grants_write_permission():
    """The base opencode.json must include ``"write": "allow"``."""
    oc_path = here() / "opencode.json"
    cfg = json.loads(oc_path.read_text(encoding="utf-8"))
    perm = cfg.get("permission", {})
    assert perm.get("write") == "allow", (
        f"opencode.json does not grant write permission; "
        f"got permission keys: {sorted(perm.keys())}"
    )


def test_sbatch_hardening_preserves_write_permission(tmp_path):
    """The inline Python in holdout_recovery_array.sbatch must not clobber write."""
    # Simulate what the sbatch inline Python does: start from the base config,
    # apply the hardening, and verify write is still present.
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

    assert perm.get("write") == "allow", (
        "sbatch hardening clobbered the write permission"
    )
