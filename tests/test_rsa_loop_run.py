"""The RSA loop's CLI (`src.rsa.loop.run`)."""

import json

import src.rsa.loop.run as rsa_run
from src.runtime import token_usage


def test_the_cli_records_agent_token_usage_in_the_results_dir(tmp_path, monkeypatch):
    """Every agent call's usage reaches <results>/token_usage.jsonl, with a summary.

    The 2026-10-06 smoke test wrote neither: the CLI never started the usage
    log, so the spend of a run was visible only in its agents' raw logs.
    """
    token_usage.reset_usage_log()

    class FakeLoop:
        def __init__(self, cfg, spawn):
            pass

        def run(self):
            token_usage.record_usage(source="rsa:candidate", backend="opencode", model="m",
                                     input_tokens=10, output_tokens=2)
            return dict(best_model="rsa_l1", standing={"rsa_l1": {}})

    monkeypatch.setattr(rsa_run, "RSALoop", FakeLoop)
    monkeypatch.setattr(rsa_run, "coding_agent_spawner", lambda **kw: None)
    results = tmp_path / "out"
    try:
        assert rsa_run.main(rsa_run.Args(results=results)) == 0
    finally:
        token_usage.reset_usage_log()
    lines = (results / "token_usage.jsonl").read_text().splitlines()
    assert [json.loads(line)["source"] for line in lines] == ["rsa:candidate"]
    assert (results / "token_usage_summary.json").exists()


def test_agents_default_to_gemini_flash_with_time_for_several_self_checks(monkeypatch, tmp_path):
    seen = {}

    class FakeLoop:
        def __init__(self, cfg, spawn):
            pass

        def run(self):
            return dict(best_model="m", standing={"m": {}})

    monkeypatch.setattr(rsa_run, "RSALoop", FakeLoop)
    monkeypatch.setattr(rsa_run, "coding_agent_spawner", lambda **kw: seen.update(kw))
    rsa_run.main(rsa_run.Args(results=tmp_path / "out"))
    assert seen["model"] == "google/gemini-3.8-flash"
    assert seen["timeout_sec"] >= 2400
