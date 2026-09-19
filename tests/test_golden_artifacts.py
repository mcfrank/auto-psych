"""Golden-artifact harness: deterministic outputs captured before the P23 refactor.

Each test asserts against a committed fixture under ``tests/golden/``. If any
assertion fails after a refactoring commit, the commit changed behaviour and
must be reverted — never fix forward.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


def _load(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


# ── 1. Recovery metrics on a fixed probability vector ──────────────────


def test_recovery_metrics():
    from src.subjective_randomness.recovery_metrics import (
        bias,
        calibration,
        kl_regret,
        rmse,
    )

    fixture = _load("metrics.json")
    q, p = fixture["q"], fixture["p"]

    assert rmse(q, p) == pytest.approx(fixture["rmse"], abs=1e-12)
    assert kl_regret(q, p) == pytest.approx(fixture["kl_regret"], abs=1e-12)
    assert bias(q, p) == pytest.approx(fixture["bias"], abs=1e-12)
    slope, intercept = calibration(q, p)
    assert slope == pytest.approx(fixture["calibration_slope"], abs=1e-12)
    assert intercept == pytest.approx(fixture["calibration_intercept"], abs=1e-12)


# ── 2. Response generation (deterministic with fixed seed) ─────────────


@pytest.mark.slow
def test_response_generation():
    from src.subjective_randomness.holdout_data import generate_responses

    fixture = _load("response_generation.json")
    rows = generate_responses(
        fixture["model_name"],
        Path("src/pipelines/outer_loop/projects/subjective_randomness/seed_models"),
        fixture["stimuli"],
        fixture["params"],
        fixture["n_participants"],
        seed=fixture["seed"],
    )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=sorted(rows[0].keys()))
    writer.writeheader()
    for r in sorted(rows, key=lambda r: (r["participant_id"], r["trial_index"])):
        writer.writerow(r)
    actual_hash = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    assert actual_hash == fixture["expected_csv_sha256"]


# ── 3. Held-out eval pool construction ─────────────────────────────────


def test_eval_pool():
    from src.subjective_randomness.holdout_eval import build_eval_stimuli

    fixture = _load("eval_pool.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        exp1 = run_root / "experiment1" / "data"
        exp1.mkdir(parents=True)
        with (exp1 / "responses.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence_a",
                    "sequence_b",
                    "participant_id",
                    "trial_index",
                    "chose_left",
                ],
            )
            w.writeheader()
            for pair in fixture["trained_pairs"]:
                w.writerow(
                    {
                        "sequence_a": pair[0],
                        "sequence_b": pair[1],
                        "participant_id": "0",
                        "trial_index": "0",
                        "chose_left": "0",
                    }
                )

        result = build_eval_stimuli(
            run_root,
            n_experiments=1,
            n_pairs=50,
            lengths=fixture["lengths"],
            seed=42,
            exhaustive=fixture["exhaustive"],
        )

    assert result["n_pool"] == fixture["expected_n_pool"]
    assert result["n_dropped"] == fixture["expected_n_dropped"]
    assert len(result["stimuli"]) == fixture["expected_n_kept"]

    content = json.dumps(result["stimuli"], sort_keys=True)
    actual_hash = hashlib.sha256(content.encode()).hexdigest()
    assert actual_hash == fixture["expected_stimuli_sha256"]


# ── 4. Leakage check on a fixture run tree ─────────────────────────────


def test_leakage_check():
    from src.subjective_randomness.leakage_audit import leakage_check

    fixture = _load("leakage_check.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir) / "run"
        run_root.mkdir()

        seed_dir = Path(
            "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
        )
        family_dir = Path("src/subjective_randomness/model_families")

        cm = run_root / "experiment1" / "cognitive_models"
        cm.mkdir(parents=True)
        (cm / "clean_model.py").write_text(
            'import pymc as pm\nimport numpy as np\n\n'
            'with pm.Model() as model:\n'
            '    beta = pm.Normal("beta", 0, 1)\n'
            '    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta))\n'
            '    obs = pm.Bernoulli("obs", p=p_left, '
            'observed=pm.Data("chose_left", [0]))\n'
        )

        ml = run_root / "experiment1" / "model_loop" / "models"
        ml.mkdir(parents=True)
        (ml / "another_model.py").write_text(
            'import pymc as pm\nimport numpy as np\n\n'
            'def compute_features(sequence_a, sequence_b):\n'
            '    return {"len_a": len(sequence_a)}\n\n'
            'with pm.Model() as model:\n'
            '    beta = pm.Normal("beta", 0, 1)\n'
            '    feat = pm.Data("len_a", [4])\n'
            '    p_left = pm.Deterministic("p_left", '
            'pm.math.sigmoid(beta * feat))\n'
            '    obs = pm.Bernoulli("obs", p=p_left, '
            'observed=pm.Data("chose_left", [0]))\n'
        )

        data_dir = run_root / "experiment1" / "data"
        data_dir.mkdir(parents=True)
        with (data_dir / "responses.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence_a",
                    "sequence_b",
                    "participant_id",
                    "trial_index",
                    "chose_left",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "sequence_a": "HHHT",
                    "sequence_b": "HTTH",
                    "participant_id": "0",
                    "trial_index": "0",
                    "chose_left": "0",
                }
            )

        result = leakage_check(
            run_root,
            fixture["gt_model"],
            seed_models_dir=seed_dir,
            n_experiments=fixture["n_experiments"],
            gt_family_dir=family_dir,
        )

    expected = fixture["expected"]
    assert result["any_identical"] == expected["any_identical"]
    assert result["any_mention"] == expected["any_mention"]
    assert result["any_value_mention"] == expected["any_value_mention"]
    assert result["any_gt_named"] == expected["any_gt_named"]
    assert result["any_csv_generating_model"] == expected["any_csv_generating_model"]
    assert result["any_manifest_gt_named"] is expected["any_manifest_gt_named"]
    assert len(result["files"]) == expected["n_files"]

    by_path = {f["path"]: f for f in result["files"]}
    assert sorted(by_path["experiment1/cognitive_models/clean_model.py"]["data_columns"]) == expected["clean_model_data_columns"]
    assert sorted(by_path["experiment1/model_loop/models/another_model.py"]["data_columns"]) == expected["another_model_data_columns"]


# ── 5. Lens schedule over 3 experiments × 2 rounds × 3 candidates ─────


def test_lens_schedule():
    from src.pipelines.inner_loop.model_zoo import _lens_index, _lens_offset

    fixture = _load("lens_schedule.json")

    for entry in fixture:
        offset = _lens_offset(
            entry["experiment"], max_iterations=2, candidate_count=3
        )
        assert offset == entry["lens_offset"], (
            f"Experiment {entry['experiment']}: expected offset {entry['lens_offset']}, got {offset}"
        )
        idx = _lens_index(offset, entry["iteration"], 3, entry["candidate_idx"], 7)
        assert idx == entry["lens_index"], (
            f"Exp {entry['experiment']} iter {entry['iteration']} cand {entry['candidate_idx']}: "
            f"expected lens {entry['lens_index']}, got {idx}"
        )


# ── 6. Import gate verdict ─────────────────────────────────────────────


def test_import_gate():
    from src.pipelines.inner_loop.import_gate import check_forbidden_imports

    fixture = _load("import_gate.json")

    allowed_result = check_forbidden_imports(fixture["allowed_source"])
    assert allowed_result == fixture["allowed_result"]

    forbidden_result = check_forbidden_imports(fixture["forbidden_source"])
    assert forbidden_result == fixture["forbidden_result"]


# ── 7. Export selection on a fixed comparison table ─────────────────────


def test_export_selection():
    from src.pipelines.inner_loop.scoring import (
        _best_exportable_model,
        _best_model,
        _unreliable_names,
    )

    fixture = _load("export_selection.json")

    assert _best_exportable_model(fixture["posterior"], fixture["comparison"]) == fixture["best_exportable_model"]
    assert _best_model(fixture["posterior"]) == fixture["best_argmax_model"]
    assert _unreliable_names(fixture["comparison"]) == fixture["unreliable_names"]
