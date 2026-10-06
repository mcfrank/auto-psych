"""Briefs, prompts and the self-check for the RSA loop's agents."""

import pandas as pd
import pytest

from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.inner_loop.model_zoo import SLOT_EXPLORE, SLOT_REFINE_CHOSEN, SLOT_REFINE_INCUMBENT
from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.loop.brief import (
    DEFAULT_RSA_LENSES,
    ZooModel,
    build_prompt,
    context_md,
    repair_note,
    write_docs,
)
from src.rsa.loop.check_candidate import check
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def live():
    return [
        ZooModel("rsa_l1_salience", "Depth-1 RSA with salience.", SEEDS / "rsa_l1_salience.py", "best"),
        ZooModel("rsa_l1", "Vanilla RSA.", SEEDS / "rsa_l1.py", "32.6 ± 7.5 nats behind the best"),
    ]


@pytest.mark.parametrize("role", [SLOT_EXPLORE, SLOT_REFINE_INCUMBENT, SLOT_REFINE_CHOSEN])
def test_each_role_gets_its_documents_and_the_handbook(tmp_path, role):
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    d = tmp_path / "cand"
    ctx = context_md(candidate_dir=d, responses_path=DEFAULT_TRIALS_CSV, round_index=0,
                     n_rounds=5, n_trials=6703, experiments=["E8_levels"])
    docs = write_docs(d, role=role, lens=DEFAULT_RSA_LENSES[0], context=ctx, live=live(),
                      pruned=[], incumbent="rsa_l1_salience", ledger=ledger)
    prompt = build_prompt(d, docs)
    assert "choice_probs(params, ctx)" in prompt
    assert "The memo Handbook" in prompt and (d / "memo_handbook.md").exists()
    assert "check_candidate" in prompt
    assert ("## attempted_hypotheses.md" in prompt) == (role == SLOT_EXPLORE)
    assert ("## refinement_menu.md" in prompt) == (role == SLOT_REFINE_CHOSEN)
    if role == SLOT_REFINE_INCUMBENT:
        assert "**rsa_l1_salience**" in docs["brief"]


def test_a_repair_note_carries_the_reason_verbatim(tmp_path):
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    reason = "contract: probabilities over objects must sum to 1 (off by 0.5)"
    docs = write_docs(tmp_path / "c", role=SLOT_EXPLORE, lens="x", context="ctx", live=live(),
                      pruned=[], incumbent="rsa_l1_salience", ledger=ledger,
                      attempt_note=repair_note(reason))
    assert reason in build_prompt(tmp_path / "c", docs)


@pytest.mark.slow
def test_the_self_check_passes_a_seed_and_fails_a_broken_copy(tmp_path):
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    responses = tmp_path / "responses.csv"
    df[df.experiment == "E8_levels"].to_csv(responses, index=False)
    good = tmp_path / "good"
    good.mkdir()
    (good / "candidate.py").write_text((SEEDS / "rsa_l1.py").read_text())
    (good / "hypothesis.md").write_text("Vanilla RSA.")
    ok, message = check(good, responses)
    assert ok, message
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "candidate.py").write_text((SEEDS / "rsa_l1.py").read_text().replace("lex: ...", "lex"))
    (bad / "hypothesis.md").write_text("Vanilla RSA.")
    ok, message = check(bad, responses)
    assert not ok and message.startswith("FAIL")
