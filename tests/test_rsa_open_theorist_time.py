"""The theorists' clock and quick fit (PI 2026-10-28, after run C: 19 of 23 Gemini 4 sessions were cut
off at 40 minutes with no candidate.py, all still exploring with fitting scripts of their own)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_rsa_open_loop_fitting import corpus, theory_path  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]


def test_the_quick_fit_finds_the_optimum_and_its_uncertainty(theory_path, corpus):  # noqa: F811
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.quick import quick_fit

    blocks, _ = corpus
    fit = quick_fit(TheoryModel(theory_path), blocks, steps=400, restarts=2)
    assert set(fit.best) >= set(TheoryModel(theory_path).params) and len(fit.log_density) == 2
    assert fit.log_density[0] >= fit.log_density[1]
    text = "\n".join(fit.lines())
    assert "Quick fit (2 starts" in text and "never how the loop scores" in text
    assert fit.problem or all(v >= 0 for v in fit.sd.values())


def test_the_quick_fit_names_a_parameter_the_data_cannot_pin_down(tmp_path, theory_path, corpus):  # noqa: F811
    """A theory with a parameter the likelihood never uses: its direction is flat."""
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.quick import quick_fit

    source = theory_path.read_text()
    assert "PARAMS = {" in source
    flat = tmp_path / "flat.py"
    flat.write_text(source.replace("PARAMS = {", 'PARAMS = {"unused": dist.Normal(0.0, 1000.0), ', 1))
    if "unused" not in TheoryModel(flat).params:
        pytest.skip("the theory's PARAMS is not a plain dict literal")
    fit = quick_fit(TheoryModel(flat), corpus[0], steps=300, restarts=1)
    assert fit.problem or fit.sd["unused"] > 50  # singular, or the prior's width: the data say nothing


def test_the_brief_asks_for_a_first_version_early_and_states_the_session_s_end(tmp_path):
    from src.rsa_open.loop.brief import DRAFT_MINUTES, context_md, time_note, write_deadline

    text = context_md(candidate_dir=tmp_path, corpus_path=tmp_path, full_corpus_path=tmp_path, admission_limit_sec=None,
                      blocks=[], data_paths={}, paradigm_notes={}, round_index=0, n_rounds=1, session_minutes=60)
    assert "ends after **60 minutes**" in text and f"within {DRAFT_MINUTES} minutes" in text and "--quick" in text
    now = time.time()
    write_deadline(tmp_path, 3600, now=now)
    assert time_note(tmp_path, now=now).startswith("Time: 60 min left") and "Finish now" not in time_note(tmp_path, now=now)
    assert "Finish now" in time_note(tmp_path, now=now + 3300)
    assert time_note(tmp_path / "none") is None


def test_the_self_check_prints_the_time_left_and_runs_the_quick_fit(tmp_path, theory_path, corpus):  # noqa: F811
    from src.rsa_open.loop.brief import write_deadline

    cdir = tmp_path / "cand"
    cdir.mkdir()
    (cdir / "candidate.py").write_text(theory_path.read_text())
    write_deadline(cdir, 1800)
    out = subprocess.run([sys.executable, "-m", "src.rsa_open.loop.check_candidate", str(cdir), "--corpus",
                          str(corpus[1]), "--full-corpus", str(corpus[1]), "--limit-minutes", "60", "--quick"],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Quick fit" in out.stdout and "min left in this session" in out.stdout


def test_the_theorists_get_an_hour_and_the_loop_states_it(tmp_path):
    from src.rsa_open.run_chain import Args

    assert Args.__dataclass_fields__["theorist_timeout_sec"].default == 3600
    for script in ("open_rehearsal.sbatch", "open_live.sbatch"):
        assert 'THEORIST_TIMEOUT_SEC:-3600' in (ROOT / "scripts/rsa/slurm" / script).read_text()
    src = (ROOT / "src/rsa_open/loop/orchestrator.py").read_text()
    assert "briefs.write_deadline(cdir, self.cfg.theorist_timeout_sec)" in src


def test_a_repair_after_a_sampling_rejection_gets_the_reparameterisation_checklist():
    from src.rsa_open.loop.brief import repair_note

    assert "Making the posterior easier to sample" in repair_note("the fit did not converge (...): R-hat 1.6")
    assert "--quick" in repair_note("fitting failed: x: the fit did not finish within 44 minutes")
    assert "Making the posterior" not in repair_note("not novel: within RMSE 0.001 of rsa_l1")


def test_unpriced_records_are_estimated_from_their_model_s_priced_ones():
    from src.rsa_open.spend import estimate_unpriced
    from src.runtime.token_usage import UsageRecord

    priced = [UsageRecord("theorist", "opencode", "google/gemini-4-argon", input_tokens=i, output_tokens=o,
                          cost_usd=1e-6 * i + 1e-5 * o) for i, o in ((1000, 100), (5000, 300), (2000, 900))]
    unpriced = [UsageRecord("designer", "opencode", "google/gemini-4-argon", input_tokens=10_000, output_tokens=1000,
                            cost_usd=0.0),
                UsageRecord("pilot", "langchain", "gemini-3.8-flash", input_tokens=500, output_tokens=50)]
    est = estimate_unpriced(priced + unpriced)
    assert est["estimated"][0]["estimated_usd"] == pytest.approx(0.02, rel=1e-3)
    assert est["unpriced_models"] == [{"model": "gemini-3.8-flash", "records": 1, "tokens": 550}]
