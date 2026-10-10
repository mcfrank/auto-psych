"""The second rehearsal's ground truth: coherent with the human data, not a seed (PI 2026-10-10)."""

import json
from pathlib import Path

import numpy as np

from src.rsa.outer import ground_truth as gt

PROMOTION = Path("data/rsa/live_seeds/promotion.json")
SWEEP = Path("data/rsa/sherlock_run2")
CHAIN = Path("data/rsa/live_seeds/chains/chain_0")


def test_the_coherent_rule_picks_the_unpromoted_model_near_the_cv_front_farthest_from_the_chain(tmp_path, monkeypatch):
    record = json.loads(PROMOTION.read_text())
    window = 120.0
    eligible = {c["key"] for c in record["candidates"] if not c["chosen"] and c["converged"] and c["cv_converged"]
                and c["cv_behind_best"] <= window}
    far = sorted(eligible)[1]
    # Stub the fits: every model predicts 0.25 except the one meant to be far.
    monkeypatch.setattr(gt, "pool_predictions", lambda folder, *a: {"seed_a": np.full((5, 4), 0.25)})
    monkeypatch.setattr(gt, "loop_fit", lambda path, name, *a, **k: name)
    monkeypatch.setattr(gt, "design_pool", lambda: list(range(5)))
    monkeypatch.setattr(gt, "posterior_mean_class_probs",
                        lambda model, fitted, pool: np.full((5, 4), 0.25) + (0.1 if fitted == far.split("/")[1] else 0))
    out = gt.main(gt.Args(promoted=tmp_path, data=tmp_path / "d.csv", cache=tmp_path, out=tmp_path / "gt.json",
                          rule="coherent", promotion=PROMOTION, sweep=SWEEP, chain=CHAIN, cv_window=window))
    assert out["ground_truth_key"] == far and set(out["candidates"]) == eligible
    assert Path(out["ground_truth_file"]).exists() and out["ground_truth"] == far.split("/")[1]
    assert not any(c["chosen"] for c in record["candidates"] if c["key"] in out["candidates"])
