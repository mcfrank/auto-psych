"""The rehearsal's yardstick (src/rsa_open/oracle.py): the hidden truth scored on a loop's own
corpus and folds, beside the theories the loop ended with and pruned."""

import json

import pytest
import yaml

from src.rsa.fit import FitSettings
from tests.test_rsa_open_design import SEEDS

QUICK = FitSettings(num_warmup=40, num_samples=40, num_chains=2, seed=0, dense_mass=True)


def _ported(name):
    from src.rsa_open.port import port

    return port((SEEDS / f"{name}.py").read_text(), name=f"{name}.py")


@pytest.mark.slow  # fits three theories and their folds
def test_the_truth_is_scored_beside_the_live_and_pruned_theories_on_the_loop_s_folds(tmp_path):
    import hashlib

    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.fitting import write_corpus
    from src.rsa_open.oracle import Args, score
    from tests.test_rsa_open_corpus import simulated_block

    truth_dir = tmp_path / "private"
    truth_dir.mkdir()
    (truth_dir / "ground_truth.py").write_text(_ported("rsa_l1_salience"))
    (truth_dir / "ground_truth.json").write_text(json.dumps({
        "theory_file": "ground_truth.py",
        "theory_sha256": hashlib.sha256((truth_dir / "ground_truth.py").read_text().encode()).hexdigest(),
        "theory_params": {}, "link_params": {}}))
    gt = TheoryModel(truth_dir / "ground_truth.py")
    loop = tmp_path / "loop"
    (loop / "models" / "pruned").mkdir(parents=True)
    (loop / "models" / "rsa_l1.py").write_text(_ported("rsa_l1"))
    (loop / "models" / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "rsa_l1"}]}))
    (loop / "models" / "pruned" / "literal_listener.py").write_text(_ported("literal_listener"))
    (loop / ".fit_cache").mkdir()
    write_corpus(loop / "corpus.json",
                 [simulated_block(gt, "E", likelihood("choice"), 90, 1, {"temperature": 1.0, "lapse": 0.05}),
                  simulated_block(gt, "exp1", likelihood("production"), 90, 2, {"temperature": 1.5, "lapse": 0.05})])
    record = score(Args(loop_dir=loop, truth_dir=truth_dir, out=tmp_path / "oracle", cv_folds=3),
                   settings=QUICK, fit_time_limit_sec=None, cv_warmup=40, cv_samples=40)
    assert set(record["theories"]) == {"rsa_l1", "literal_listener", "ground_truth"}
    assert record["ended_live"] == ["rsa_l1"] and record["blocks"] == ["E", "exp1"]
    truth = record["theories"]["ground_truth"]["by_block"]["exp1"]
    assert truth["minus_truth"] == 0.0
    md = (tmp_path / "oracle" / "oracle.md").read_text()
    assert "| ground_truth" in md and "| truth |" in md and "| pruned |" in md
    assert any((loop / ".fit_cache").glob("ground_truth.*.nc"))  # the loop's cache, linked


def test_the_oracle_is_kept_out_of_the_agents_tree():
    from pathlib import Path

    exclude = (Path(__file__).resolve().parents[1] / "scripts/rsa/slurm/agent_tree.exclude").read_text().split()
    assert "/src/rsa_open/oracle.py" in exclude
