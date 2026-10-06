"""The RSA loop end to end with a scripted fake agent (no API calls)."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.fit import FitSettings
from src.rsa.loop.orchestrator import LoopConfig, RSALoop
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def write(cdir, source, hypothesis, name=None):
    (cdir / "candidate.py").write_text(source)
    (cdir / "hypothesis.md").write_text(hypothesis)
    if name:
        (cdir / "model_name.txt").write_text(name + "\n")


def fake_agent(cdir, prompt):
    """Slot 1 proposes depth-2 RSA; slot 2 breaks memo then repairs it;
    slot 3 never writes a file."""
    slot = cdir.name
    if slot.startswith("candidate_1"):
        write(cdir, (SEEDS / "rsa_l2.py").read_text(), "Listeners reason at depth 2.", "deeper_listener")
    elif slot == "candidate_2":
        broken = (SEEDS / "rsa_l1_salience.py").read_text().replace("{EPS}", "EPS")
        write(cdir, broken, "A salience prior at the top level.", "salience_listener")
    elif slot == "candidate_2_repair_1":
        assert "REPAIR" in prompt and "MemoError" in prompt
        write(cdir, (SEEDS / "rsa_l1_salience.py").read_text(), "A salience prior at the top level.",
              "salience_listener")
    return True


@pytest.fixture
def loop(tmp_path):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for name in ("literal_listener", "rsa_l1"):
        shutil.copyfile(SEEDS / f"{name}.py", seeds / f"{name}.py")
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [
        {"name": "literal_listener", "rationale": "Literal listener."},
        {"name": "rsa_l1", "rationale": "Depth-1 RSA."},
    ]}))
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    responses = tmp_path / "trials.csv"
    df[df.experiment.isin(["E8_levels", "E5_baserate"])].to_csv(responses, index=False)
    cfg = LoopConfig(
        responses_path=responses, seed_models_dir=seeds, results_dir=tmp_path / "model_loop",
        max_iterations=1, candidate_count=3,
        settings=FitSettings(num_warmup=300, num_samples=300, num_chains=2),
        fit_time_limit_sec=None,
    )
    return RSALoop(cfg, fake_agent)


def test_a_round_admits_repairs_retries_and_exports(loop):
    final = loop.run()
    out = loop.dir
    ledger = [json.loads(line) for line in (out / "attempted_hypotheses.jsonl").read_text().splitlines()]
    by = {(e["name"], e["outcome"]) for e in ledger}
    assert ("deeper_listener", "admitted") in by
    assert ("salience_listener", "rejected") in by and ("salience_listener_2", "admitted") in by
    assert any(e["outcome"] == "rejected" and "no candidate.py" in e["detail"] for e in ledger)
    assert (out / "round_1" / "candidate_3_retry_1").exists()
    # Seeds, the deeper model and the repaired salience model were live before pruning.
    history = json.loads((out / "history.json").read_text())
    assert [h["step"] for h in history] == [0, 1, 2]
    assert set(history[1]["standing"]) == {"literal_listener", "rsa_l1", "deeper_listener", "salience_listener_2"}
    # The literal listener loses by far more than 2 clustered SEs and is pruned at the end.
    assert "literal_listener" not in final["standing"]
    assert (out / "models" / "pruned" / "literal_listener.py").exists()
    assert (out / "best_model.py").exists()
    export = json.loads((out / "export.json").read_text())
    assert export["best_model"] == final["best_model"]
    report = (out / "report.html").read_text()
    assert "__BUNDLE_JSON__" not in report and "salience_listener_2" in report
    bundle = json.loads((out / "report.bundle.json").read_text())
    assert len(bundle["timeline"]["history"]) == 3
