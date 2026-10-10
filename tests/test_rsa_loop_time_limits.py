"""Only a candidate's admission fit is time-limited (the 2026-10-09 rehearsal
died when a reference model's fit passed 30 minutes)."""

import math

import pandas as pd

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.loop.orchestrator import LoopConfig, RSALoop
from src.runtime.config import PROJECT_ASSETS_DIR


def test_models_in_play_fit_without_a_limit_and_candidates_with_one(tmp_path):
    data = tmp_path / "r.csv"
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    df[df.experiment.isin(["E8_levels", "E5_baserate"])].to_csv(data, index=False)
    loop = RSALoop(LoopConfig(responses_path=data, seed_models_dir=PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models",
                              results_dir=tmp_path / "loop", fit_time_limit_sec=1800, cv_folds=2), spawn=None)
    loop._prepare()
    assert loop.gate_cfg.time_limit_sec == 1800
    assert math.isinf(loop.live_cfg.time_limit_sec)
