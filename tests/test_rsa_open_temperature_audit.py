"""The audit of E's fixed temperature (src/rsa_open/temperature_audit.py): every theory fitted with
the temperature fixed and free, and the difference in predictive density between them."""

import json

import numpy as np
import pytest

from src.rsa.fit import FitSettings
from tests.test_rsa_open_loop_orchestrator import ported

SETTINGS = FitSettings(num_warmup=150, num_samples=150, num_chains=2, seed=0, dense_mass=True)


@pytest.mark.slow  # four small fits
def test_each_theory_is_fitted_with_the_temperature_fixed_and_free(tmp_path):
    from src.rsa_open.corpus import LITERATURE_FIXED
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.links import likelihood
    from src.rsa_open.temperature_audit import audit
    from tests.test_rsa_open_corpus import simulated_block

    theories = {}
    for name in ("literal_listener", "rsa_l1"):
        theories[name] = tmp_path / f"{name}.py"
        theories[name].write_text(ported(name))
    gt = tmp_path / "gt.py"
    gt.write_text(ported("rsa_l1_salience"))
    E = simulated_block(TheoryModel(gt), "E", likelihood("choice"), 200, 1, {"temperature": 2.0, "lapse": 0.05})
    from dataclasses import replace

    rows = audit(theories, replace(E, fixed=LITERATURE_FIXED), tmp_path / "audit", settings=SETTINGS,
                 time_limit_sec=None, workers=2)
    assert [r["theory"] for r in rows] == ["literal_listener", "rsa_l1"]
    for r in rows:
        assert "temperature@E" not in r["fixed"]["means"] and "temperature@E" in r["free"]["means"]
        assert np.isfinite(r["elpd_free_minus_fixed"]) and r["dse_clustered"] >= 0
    assert json.loads((tmp_path / "audit" / "audit.json").read_text()) == rows
    assert "| rsa_l1 |" in (tmp_path / "audit" / "audit.md").read_text()
