"""Agent write access is unconditional: no agent session may be denied access
to its own working directory, and a denial can never pass quietly.

The diagnosis (P34), from opencode 1.18.8's bundled source and the archived
sweep logs (``sweep_rerun/run3/motif_stack``):

* ``opencode run`` roots its session at ``process.env.PWD ?? process.cwd()``.
  ``subprocess.Popen(cwd=...)`` changes the child's directory but leaves the
  inherited ``PWD`` untouched, so an agent launched by the holdout harness
  (whose shell had ``cd``'d into ``harness_repo``) got a session rooted at
  ``harness_repo``. Its own candidate directory under the agent tree was then
  "external", asked for, and auto-rejected by the non-interactive ``run``.
* The permission config is loaded from that session directory too, so the
  grants the sbatch wrote into the agent tree's ``opencode.json`` were never
  the ones in force.

The launcher therefore pins ``PWD`` to the real cwd, writes any grant an
allowed directory outside the cwd needs into the cwd's ``opencode.json``, and
treats an ``auto-rejecting`` line in the agent's log as a misconfiguration to
raise on — not as one unlucky candidate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.runtime import coding_agent

# Verbatim from sweep_rerun/run3/motif_stack, experiment3/iter_1/candidate_2.
ARCHIVED_DENIAL_LINE = (
    "\x1b[93m\x1b[1m! \x1b[0mpermission requested: external_directory "
    "(/scratch/users/benpry/auto-psych/consolidation_2026_09/sweep_rerun/run3/"
    "motif_stack/repo/_runs/motif_stack/experiment3/model_loop/iter_1/candidate_2/*); "
    "auto-rejecting"
)

# A stand-in `opencode` binary. It must be Python, not bash: bash resets an
# inherited PWD that does not name the current directory, which would hide
# exactly the bug these tests guard against.
_FAKE_OPENCODE_REPORT_ENV = """
import json, os
cfg = os.path.join(os.getcwd(), "opencode.json")
info = {
    "PWD": os.environ.get("PWD"),
    "XDG_DATA_HOME": os.environ.get("XDG_DATA_HOME"),
    "cwd": os.getcwd(),
    "opencode_json": open(cfg).read() if os.path.exists(cfg) else None,
}
print(json.dumps({"type": "text", "part": {"text": json.dumps(info)}}))
print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1,
      "reasoning": 0, "cache": {"read": 0, "write": 0}}, "cost": 0.0}}))
"""

# Denies FAKE_DENIED_DIR (set by each test): the agent's own directory, or one
# outside everything it was given.
_FAKE_OPENCODE_DENIED = """
import json, os
print("! permission requested: external_directory (" + os.environ["FAKE_DENIED_DIR"] + "/*); auto-rejecting")
print(json.dumps({"type": "text", "part": {"text": "carried on"}}))
"""


def _install_fake_opencode(tmp_path: Path, monkeypatch, body: str) -> None:
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "opencode"
    script.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def _agent_tree(tmp_path: Path) -> Path:
    """A cwd for the agent with the repo's kind of opencode.json in it."""
    tree = tmp_path / "agent_tree"
    tree.mkdir()
    (tree / "opencode.json").write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "permission": {
                    "edit": "allow",
                    "bash": "allow",
                    "external_directory": {"/tmp/**": "allow"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tree


def _run(tmp_path: Path, tree: Path, *, allowed_dirs=(), env=None) -> dict:
    agent_dir = tmp_path / "model_loop" / "iter_0" / "candidate_0"
    agent_dir.mkdir(parents=True)
    ok, text = coding_agent.run_coding_agent(
        "hello",
        cwd=tree,
        log_path=agent_dir / "agent.jsonl",
        allowed_dirs=list(allowed_dirs),
        backend="opencode",
        timeout_secs=30,
        env=env,
        on_summary=None,
    )
    assert ok, text
    report = json.loads(text)
    report["agent_dir"] = agent_dir
    return report


# --- the session directory is the agent's real cwd ---------------------------


def test_child_pwd_is_pinned_to_the_agent_cwd(tmp_path, monkeypatch):
    _install_fake_opencode(tmp_path, monkeypatch, _FAKE_OPENCODE_REPORT_ENV)
    tree = _agent_tree(tmp_path)
    # The launching process sits somewhere else, as the harness does after
    # `cd "$HARNESS_REPO"`.
    monkeypatch.setenv("PWD", str(tmp_path / "somewhere_else"))
    report = _run(tmp_path, tree)
    assert report["PWD"] == str(tree.resolve()) == report["cwd"]


def test_child_environment_pins_pwd_for_every_backend(tmp_path):
    for backend in ("claude", "codex", "opencode"):
        env = coding_agent.child_environment(
            backend=backend,
            cwd=tmp_path,
            log_path=tmp_path / "logs" / "agent.jsonl",
            env={"PWD": "/stale", "PATH": "/usr/bin"},
        )
        assert env["PWD"] == str(tmp_path.resolve()), backend
        assert env["PATH"] == "/usr/bin", "the caller's environment is kept"


# --- a private opencode store per spawned agent -------------------------------


def test_opencode_agent_gets_a_private_data_home_under_its_own_dir(
    tmp_path, monkeypatch
):
    _install_fake_opencode(tmp_path, monkeypatch, _FAKE_OPENCODE_REPORT_ENV)
    tree = _agent_tree(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "shared_xdg"))
    report = _run(tmp_path, tree)
    private = Path(report["XDG_DATA_HOME"])
    assert private == coding_agent.agent_data_home(report["agent_dir"] / "agent.jsonl")
    assert private.parent == report["agent_dir"], "under the agent's own directory"
    assert private.is_dir()


def test_private_data_home_is_only_for_opencode(tmp_path):
    log_path = tmp_path / "candidate_0" / "agent.jsonl"
    shared = {"XDG_DATA_HOME": "/shared"}
    assert (
        coding_agent.child_environment(
            backend="claude", cwd=tmp_path, log_path=log_path, env=shared
        )["XDG_DATA_HOME"]
        == "/shared"
    )
    assert coding_agent.child_environment(
        backend="opencode", cwd=tmp_path, log_path=log_path, env=shared
    )["XDG_DATA_HOME"] == str(coding_agent.agent_data_home(log_path))


def test_private_data_home_links_the_inherited_credentials(tmp_path):
    shared = tmp_path / "shared_xdg"
    auth = shared / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text('{"google": {"type": "api", "key": "k"}}', encoding="utf-8")
    log_path = tmp_path / "candidate_0" / "agent.jsonl"
    env = coding_agent.child_environment(
        backend="opencode",
        cwd=tmp_path,
        log_path=log_path,
        env={"XDG_DATA_HOME": str(shared)},
    )
    linked = Path(env["XDG_DATA_HOME"]) / "opencode" / "auth.json"
    assert linked.is_symlink() and linked.resolve() == auth.resolve()
    # Idempotent: a retry in the same directory must not trip over the link.
    coding_agent.child_environment(
        backend="opencode",
        cwd=tmp_path,
        log_path=log_path,
        env={"XDG_DATA_HOME": str(shared)},
    )
    assert linked.resolve() == auth.resolve()


# --- grants for allowed directories outside the cwd ---------------------------


def test_grants_are_written_for_dirs_outside_the_cwd_and_existing_ones_kept(tmp_path):
    tree = _agent_tree(tmp_path)
    inside = tree / "_runs" / "exp1" / "model_loop" / "models"
    outside = tmp_path / "scratch" / "run1" / "responses"
    added = coding_agent.ensure_opencode_external_grants(tree, [inside, outside])
    assert added == [f"{outside.resolve()}/**", f"{outside.resolve()}/*"]
    cfg = json.loads((tree / "opencode.json").read_text(encoding="utf-8"))
    ext = cfg["permission"]["external_directory"]
    assert ext[f"{outside.resolve()}/**"] == "allow"
    assert ext[f"{outside.resolve()}/*"] == "allow"
    assert ext["/tmp/**"] == "allow", "existing grants are preserved"
    assert cfg["permission"]["edit"] == "allow", "the rest of the config is untouched"
    assert not any(str(inside) in key for key in ext), "inside the cwd needs no grant"


def test_grants_leave_the_config_untouched_when_every_dir_is_inside(tmp_path):
    tree = _agent_tree(tmp_path)
    before = (tree / "opencode.json").read_bytes()
    added = coding_agent.ensure_opencode_external_grants(
        tree, [tree / "_runs" / "a", tree / "_runs" / "b"]
    )
    assert added == []
    assert (tree / "opencode.json").read_bytes() == before


def test_grants_raise_when_the_cwd_has_no_opencode_json(tmp_path):
    tree = tmp_path / "bare"
    tree.mkdir()
    with pytest.raises(FileNotFoundError, match="opencode.json"):
        coding_agent.ensure_opencode_external_grants(tree, [tmp_path / "elsewhere"])


def test_run_coding_agent_writes_the_grants_before_spawning(tmp_path, monkeypatch):
    _install_fake_opencode(tmp_path, monkeypatch, _FAKE_OPENCODE_REPORT_ENV)
    tree = _agent_tree(tmp_path)
    outside = tmp_path / "scratch" / "models"
    report = _run(tmp_path, tree, allowed_dirs=[tree / "_runs", outside])
    seen = json.loads(report["opencode_json"])["permission"]["external_directory"]
    assert seen[f"{outside.resolve()}/**"] == "allow"
    assert seen["/tmp/**"] == "allow"


# --- a denial is a misconfiguration, never a per-slot rejection ---------------


# The archived line denied the agent its own candidate directory: the session
# was rooted in the wrong tree. That is a misconfigured launch.
ARCHIVED_AGENT_TREE = Path(
    "/scratch/users/benpry/auto-psych/consolidation_2026_09/sweep_rerun/run3/motif_stack/repo"
)


def test_a_denial_of_the_agents_own_directory_raises(tmp_path):
    log = tmp_path / "agent.jsonl"
    log.write_text(
        json.dumps({"type": "step_start"}) + "\n" + ARCHIVED_DENIAL_LINE + "\n",
        encoding="utf-8",
    )
    with pytest.raises(coding_agent.AgentPermissionDenied) as info:
        coding_agent.check_for_permission_denials(log, own_dirs=[ARCHIVED_AGENT_TREE])
    assert "external_directory" in str(info.value)
    assert "candidate_2" in str(info.value)
    assert str(log) in str(info.value)


def test_a_clean_log_does_not_raise(tmp_path):
    log = tmp_path / "agent.jsonl"
    log.write_text(
        json.dumps({"type": "text", "part": {"text": "wrote candidate.py"}}) + "\n",
        encoding="utf-8",
    )
    assert coding_agent.check_for_permission_denials(log, own_dirs=[tmp_path]) == []


def test_an_agent_displaying_the_denial_text_is_not_a_denial(tmp_path):
    """Tool calls and their output are JSON events in the log. An agent that
    cat-s coding_agent.py (or its own earlier log) shows the signature inside
    one; that used to raise AgentPermissionDenied and kill the cell."""
    shown = 'PERMISSION_DENIAL_SIGNATURE = "auto-rejecting"\n' + ARCHIVED_DENIAL_LINE
    log = tmp_path / "agent.jsonl"
    log.write_text(
        json.dumps(
            {"type": "tool_use", "part": {"tool": "bash", "state": {"output": shown}}}
        )
        + "\n"
        + json.dumps(
            {"type": "text", "part": {"text": "auto-rejecting is opencode's word"}}
        )
        + "\n",
        encoding="utf-8",
    )
    assert coding_agent.check_for_permission_denials(log, own_dirs=[tmp_path]) == []


def test_a_denial_outside_the_agents_directories_is_returned_not_raised(tmp_path):
    """The agent reached outside its tree and was refused: that is the refusal
    working, not a misconfiguration. The caller reports it and carries on."""
    tree = tmp_path / "run_a" / "repo"
    outside = tmp_path / "run_b" / "repo"
    log = tmp_path / "agent.jsonl"
    log.write_text(
        f"! permission requested: external_directory ({outside}/*); auto-rejecting\n",
        encoding="utf-8",
    )
    refused = coding_agent.check_for_permission_denials(log, own_dirs=[tree])
    assert refused == [str(outside)]


def test_run_coding_agent_raises_when_denied_its_own_directory(tmp_path, monkeypatch):
    _install_fake_opencode(tmp_path, monkeypatch, _FAKE_OPENCODE_DENIED)
    tree = _agent_tree(tmp_path)
    candidate_dir = tree / "candidate_0"
    monkeypatch.setenv("FAKE_DENIED_DIR", str(candidate_dir))
    log_path = candidate_dir / "agent.jsonl"
    with pytest.raises(coding_agent.AgentPermissionDenied):
        coding_agent.run_coding_agent(
            "hello",
            cwd=tree,
            log_path=log_path,
            backend="opencode",
            allowed_dirs=[candidate_dir],
            timeout_secs=30,
            on_summary=None,
        )
    assert "auto-rejecting" in log_path.read_text(encoding="utf-8")


def test_an_agent_refused_a_directory_outside_its_own_carries_on(tmp_path, monkeypatch):
    _install_fake_opencode(tmp_path, monkeypatch, _FAKE_OPENCODE_DENIED)
    tree = _agent_tree(tmp_path)
    candidate_dir = tree / "candidate_0"
    monkeypatch.setenv("FAKE_DENIED_DIR", str(tmp_path / "another_run"))
    summaries = []
    success, text = coding_agent.run_coding_agent(
        "hello",
        cwd=tree,
        log_path=candidate_dir / "agent.jsonl",
        backend="opencode",
        allowed_dirs=[candidate_dir],
        timeout_secs=30,
        on_summary=summaries.append,
    )
    assert success
    assert "carried on" in text
    assert any("refused" in s and "another_run" in s for s in summaries)


def test_critique_round_does_not_swallow_a_permission_denial(tmp_path, monkeypatch):
    from src.pipelines.inner_loop import critique_round, scoring

    monkeypatch.setattr(
        scoring, "_best_exportable_model", lambda posterior, comparison: "seed"
    )

    def denied(*args, **kwargs):
        raise coding_agent.AgentPermissionDenied("denied")

    monkeypatch.setattr(critique_round, "_spawn_critique_agent", denied)
    with pytest.raises(coding_agent.AgentPermissionDenied):
        critique_round._run_critique_round(
            tmp_path / "iter_0",
            responses_path=tmp_path / "responses.csv",
            models_dir=tmp_path / "models",
            posterior={},
            comparison={},
            cache_dir=None,
            fit_kwargs={},
            n_proposals=1,
            significance_alpha=0.05,
            n_replicates=1,
            agent_timeout_sec=1,
            backend="opencode",
        )


def test_opencode_agents_get_a_long_shell_timeout_unless_one_is_set(tmp_path):
    log_path = tmp_path / "candidate_0" / "agent.jsonl"
    name = coding_agent.OPENCODE_BASH_TIMEOUT_ENV
    env = coding_agent.child_environment(backend="opencode", cwd=tmp_path, log_path=log_path, env={})
    assert env[name] == str(coding_agent.OPENCODE_BASH_TIMEOUT_MS)
    assert int(env[name]) >= 10 * 60 * 1000  # a self-check takes up to ~10 min
    kept = coding_agent.child_environment(
        backend="opencode", cwd=tmp_path, log_path=log_path, env={name: "5000"}
    )
    assert kept[name] == "5000"
    assert name not in coding_agent.child_environment(
        backend="claude", cwd=tmp_path, log_path=log_path, env={}
    )


def test_opencode_snapshots_are_off_and_survive_the_grant_rewrite(tmp_path):
    """opencode's per-agent git snapshot of its working tree was ~1 GB per
    agent in the RSA smoke test; the repo config turns it off, and adding
    directory grants must not drop the setting."""
    import json

    from src.runtime.config import REPO_ROOT

    assert json.loads((REPO_ROOT / "opencode.json").read_text())["snapshot"] is False
    root = tmp_path / "tree"
    root.mkdir()
    (root / "opencode.json").write_text((REPO_ROOT / "opencode.json").read_text())
    (root / ".here").write_text("")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    coding_agent.ensure_opencode_external_grants(root, [outside])
    assert json.loads((root / "opencode.json").read_text())["snapshot"] is False
