"""The novelty gate measures novelty on a broad, loop-generated stimulus pool.

Measured on the 64 training stimuli at 0.02 RMSE, the gate's 23 archived
rejection margins in the September 2026 sweep were bimodal: about five genuine
re-skins clustered at ~0 (two predicting *identically*) and about eighteen
spread evenly from 0.006 up to the threshold — what distinct mechanisms that
happen to agree on 64 points look like. The gate now compares posterior-mean
``p_left`` on a pool the loop generates from its own seed, never the recovery
harness's eval pool (the loop must not select models on the stimuli it is
later scored against), at a threshold of 0.002 that separates the re-skin
cluster from everything else.

MCMC is stubbed throughout: fitted models are fakes whose ``predict_p_left``
is a function of the rows they are asked about, which is exactly what lets a
test distinguish "measured on the training stimuli" from "measured on the pool".
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.models.data_binding import MissingStimulusColumns
from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    NOVELTY_POOL_FILENAME,
    NOVELTY_POOL_LENGTHS,
    NOVELTY_POOL_N_PAIRS,
    NOVELTY_POOL_SEED,
    _admit_candidate,
    _min_prediction_rmse,
    novelty_pool_rows,
)
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.subjective_randomness.stimulus_design import (
    enumerate_all_pairs,
    generate_candidate_pool,
)
from tests.inner_loop_fixtures import write_responses, write_seed_models

# The recovery harness's default ``eval_pool.seed`` (holdout_recovery.py,
# ``run_holdout_recovery_from_config``); the loop's pool must not be that sample.
HARNESS_EVAL_POOL_SEED = 11


def _unordered(row):
    return tuple(sorted((row["sequence_a"], row["sequence_b"])))


# ── The pool ──────────────────────────────────────────────────────────


def test_novelty_pool_is_loop_generated_and_not_the_eval_pool():
    pool = novelty_pool_rows()

    # Deterministic from the loop's own seed, raw stimulus rows only.
    assert pool == novelty_pool_rows()
    assert len(pool) == NOVELTY_POOL_N_PAIRS
    assert all(set(row) == {"sequence_a", "sequence_b"} for row in pool)
    assert all(set(row["sequence_a"] + row["sequence_b"]) <= {"H", "T"} for row in pool)
    assert all(len(row["sequence_a"]) == len(row["sequence_b"]) for row in pool)
    assert {len(row["sequence_a"]) for row in pool} == set(NOVELTY_POOL_LENGTHS)
    assert len({_unordered(row) for row in pool}) == len(pool)

    # Not the recovery harness's eval pool: neither its sampled pool at the
    # harness's default seed nor its exhaustive pool over lengths 1–8.
    assert NOVELTY_POOL_SEED != HARNESS_EVAL_POOL_SEED
    harness_sample = generate_candidate_pool(
        NOVELTY_POOL_N_PAIRS, lengths=NOVELTY_POOL_LENGTHS, seed=HARNESS_EVAL_POOL_SEED
    )
    assert {_unordered(r) for r in pool} != {_unordered(r) for r in harness_sample}
    exhaustive = enumerate_all_pairs(range(1, 9), same_length_only=True)
    assert len(pool) < len(exhaustive)


def test_default_threshold_is_recalibrated_to_the_re_skin_cluster():
    assert DEFAULT_NOVELTY_RMSE_THRESHOLD == 0.002


# ── Fakes: fits whose predictions depend on the rows asked about ──────


class _FakeFitted:
    """A fitted model whose ``p_left`` is ``predict(row)`` for each row bound."""

    def __init__(self, predict, *, inputs=("chose_left",), needs=()):
        self._predict = predict
        self.model = SimpleNamespace(inputs=list(inputs), needs=list(needs))

    def predict_p_left(self, stim_data, **kwargs):
        return np.asarray([self._predict(row) for row in stim_data["rows"]])


def _fake_make_stim_data(model, rows):
    """Pass the rows through; a model with ``needs`` cannot bind bare stimuli."""
    if model.needs:
        raise MissingStimulusColumns(model.needs, list(rows[0].keys()))
    return {"rows": list(rows)}


def _stub_prediction_plumbing(monkeypatch, fitted_by_name):
    # The stub fit is not a real trace: pass the convergence gate.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(
        model_zoo, "fit_model", lambda name, *a, **k: fitted_by_name[name]
    )
    monkeypatch.setattr(model_zoo, "make_stim_data", _fake_make_stim_data)
    monkeypatch.setattr(model_zoo, "pm_data_inputs", lambda model: model.inputs)


def _stub_admission_gates(monkeypatch):
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, models_dir: object())
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -10.0)


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]}
        ),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text("# model\n", encoding="utf-8")
    return models_dir


TRAINING_STIMULI = [
    {"sequence_a": "HHTT", "sequence_b": "HTHT"},
    {"sequence_a": "HHHTT", "sequence_b": "HTHTT"},
]
TRAINED_PAIRS = {_unordered(row) for row in TRAINING_STIMULI}


def _write_responses(tmp_path, stimuli=TRAINING_STIMULI, participant_ids=(0,)):
    path = tmp_path / "responses.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_a",
                "sequence_b",
                "participant_id",
                "trial_index",
                "chose_left",
            ],
        )
        writer.writeheader()
        for pid in participant_ids:
            for i, row in enumerate(stimuli):
                writer.writerow(
                    {**row, "participant_id": pid, "trial_index": i, "chose_left": 1}
                )
    return path


def _candidate(tmp_path):
    cand_dir = tmp_path / "candidate_0"
    cand_dir.mkdir(exist_ok=True)
    (cand_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    (cand_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
    return cand_dir / "candidate.py"


def _agrees_on_training_only(row):
    """0.5 on the training stimuli (like the seed), 0.9 everywhere else."""
    return 0.5 if _unordered(row) in TRAINED_PAIRS else 0.9


# ── Where the RMSE is measured ────────────────────────────────────────


def test_min_prediction_rmse_measures_on_the_pool_not_the_training_stimuli(
    tmp_path, monkeypatch
):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "candidate_x": _FakeFitted(_agrees_on_training_only),
        },
    )

    on_training = _min_prediction_rmse(
        "candidate_x", models_dir, responses, pool_rows=TRAINING_STIMULI
    )
    on_pool = _min_prediction_rmse(
        "candidate_x", models_dir, responses, pool_rows=novelty_pool_rows()
    )

    assert on_training == ("seed_a", pytest.approx(0.0))
    assert on_pool[0] == "seed_a"
    assert on_pool[1] > 0.1


def test_pool_rows_are_bound_with_a_dummy_response_column(tmp_path, monkeypatch):
    """A bare stimulus row carries no response; the observed container is
    filled with dummies (unused by ``p_left``), as the evaluation does."""
    models_dir = _models_dir(tmp_path, [])
    responses = _write_responses(tmp_path)
    seen = []

    def remember(row):
        seen.append(row)
        return 0.5

    _stub_prediction_plumbing(monkeypatch, {"candidate_x": _FakeFitted(remember)})
    _min_prediction_rmse(
        "candidate_x", models_dir, responses, pool_rows=novelty_pool_rows()[:3]
    )
    assert len(seen) == 3
    assert all(set(row) == {"sequence_a", "sequence_b", "chose_left"} for row in seen)
    assert all(row["chose_left"] == 0 for row in seen)


def test_participant_random_effect_is_marginalised_over_training_participants(
    tmp_path, monkeypatch
):
    """A model indexing ``participant_id`` has only per-participant ``p_left``:
    each pool row is predicted as every participant the model was fit on and
    averaged, so it is compared population-level like every other model."""
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path, participant_ids=(3, 7))
    seen_pids = []

    def per_participant(row):
        seen_pids.append(row["participant_id"])
        return 0.2 if row["participant_id"] == 3 else 0.8

    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "hierarchical": _FakeFitted(
                per_participant, inputs=("participant_id", "chose_left")
            ),
        },
    )
    pool = novelty_pool_rows()[:5]
    nearest, rmse = _min_prediction_rmse(
        "hierarchical", models_dir, responses, pool_rows=pool
    )
    assert nearest == "seed_a"
    assert rmse == pytest.approx(0.0)
    assert sorted(set(seen_pids)) == [3, 7]
    assert len(seen_pids) == 2 * len(pool)


def test_candidate_that_needs_response_row_columns_is_rejected_not_crashed(
    tmp_path, monkeypatch, capsys
):
    """A candidate binding ``trial_index`` cannot be evaluated on a stimulus
    pool (nor by the recovery evaluation); admission records that as the
    rejection reason instead of aborting the round."""
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "needs_trial": _FakeFitted(lambda row: 0.5, needs=("trial_index",)),
        },
    )
    assert not _admit_candidate(
        _candidate(tmp_path), models_dir, "needs_trial", responses
    )
    out = capsys.readouterr().out
    assert "[reject] needs_trial" in out
    assert "trial_index" in out
    assert not (models_dir / "needs_trial.py").exists()


# A prepare_observed model that reads participant_id straight off each row,
# without exposing a participant_id pm.Data container. The novelty pool's
# marginalisation keys off that container, so this model is handed bare
# stimulus rows. GPT-6 Luna wrote one (session_order_side_drift) and the
# resulting KeyError took down a whole holdout cell 23 minutes in, because the
# tests above stub make_stim_data and never exercise the real hook path.
_HOOK_READS_PARTICIPANT = """
import numpy as np
import pymc as pm


def prepare_observed(rows):
    participant = np.array([int(row["participant_id"]) for row in rows])
    return {
        "odd_participant": (participant % 2).astype("float64"),
        "chose_left": np.array([int(row["chose_left"]) for row in rows], dtype="int64"),
    }


with pm.Model() as model:
    odd_participant = pm.Data("odd_participant", np.zeros(2))
    chose_left = pm.Data("chose_left", np.zeros(2, dtype="int64"))
    weight = pm.Normal("weight", 0.0, 1.0)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(weight * odd_participant))
    pm.Bernoulli("obs", p=p_left, observed=chose_left)
"""


def test_hook_reading_participant_id_is_rejected_not_crashed(
    tmp_path, monkeypatch, capsys
):
    """The real binding layer, not a stub: the candidate is a recorded
    rejection naming participant_id, and admission returns normally."""
    from src.models.model_loading import load_pymc_model

    hook_dir = tmp_path / "hook_model"
    hook_dir.mkdir()
    (hook_dir / "reads_participant.py").write_text(
        _HOOK_READS_PARTICIPANT, encoding="utf-8"
    )
    real_model = load_pymc_model("reads_participant", hook_dir)

    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path, participant_ids=(0, 1))
    _stub_admission_gates(monkeypatch)
    # Only the MCMC is faked; make_stim_data and pm_data_inputs are real.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(
        model_zoo,
        "fit_model",
        lambda name, *a, **k: SimpleNamespace(
            model=real_model, predict_p_left=lambda stim_data, **kw: np.full(1, 0.5)
        ),
    )

    admitted = _admit_candidate(
        _candidate(tmp_path), models_dir, "reads_participant", responses
    )

    assert not admitted
    out = capsys.readouterr().out
    assert "[reject] reads_participant" in out
    assert "participant_id" in out
    assert not (models_dir / "reads_participant.py").exists()


def test_an_admitted_model_that_cannot_bind_the_pool_is_a_loud_error(
    tmp_path, monkeypatch
):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5, needs=("trial_index",)),
            "candidate_x": _FakeFitted(lambda row: 0.5),
        },
    )
    with pytest.raises(RuntimeError, match="seed_a"):
        _min_prediction_rmse(
            "candidate_x", models_dir, responses, pool_rows=novelty_pool_rows()[:3]
        )


# ── Admission under the recalibrated gate ─────────────────────────────


def test_candidate_predicting_identically_to_an_existing_model_is_rejected(
    tmp_path, monkeypatch, capsys
):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.6),
            "re_skin": _FakeFitted(lambda row: 0.6),
        },
    )

    assert not _admit_candidate(_candidate(tmp_path), models_dir, "re_skin", responses)

    out = capsys.readouterr().out
    assert "predicts like existing model 'seed_a'" in out
    assert not (models_dir / "re_skin.py").exists()
    manifest = yaml.safe_load(
        (models_dir / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    assert [m["name"] for m in manifest["models"]] == ["seed_a"]


def test_model_differing_only_off_the_training_stimuli_is_admitted(
    tmp_path, monkeypatch
):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "off_training": _FakeFitted(_agrees_on_training_only),
        },
    )

    assert _admit_candidate(_candidate(tmp_path), models_dir, "off_training", responses)
    assert (models_dir / "off_training.py").exists()


@pytest.mark.parametrize(
    "offset, admitted",
    [(0.0019, False), (0.0021, True)],
    ids=["just-inside-the-re-skin-cluster", "just-outside-it"],
)
def test_gate_threshold_separates_the_re_skin_cluster(
    tmp_path, monkeypatch, offset, admitted
):
    """A constant offset of ``offset`` in ``p_left`` is an RMSE of exactly
    ``offset`` on any pool: the threshold is compared to the unrounded value."""
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "shifted": _FakeFitted(lambda row: 0.5 + offset),
        },
    )
    assert (
        _admit_candidate(_candidate(tmp_path), models_dir, "shifted", responses)
        is admitted
    )


# ── The orchestrator generates the pool once and records it ───────────


def _patch_scoring(monkeypatch):
    """``model_a`` always wins; nothing is pruned; admission gates are stubbed
    except the novelty gate, which the test under it replaces."""

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        names = model_zoo._manifest_names(models_dir)
        return {
            "posteriors": {n: (1.0 if n == "model_a" else 0.0) for n in names},
            "elpd_loo": {n: (-10.0 if n == "model_a" else -11.0) for n in names},
            "n_trials": 2,
        }

    def fake_compare(responses_path, models_dir, **kwargs):
        return {
            n: {
                "rank": rank,
                "elpd_loo": -10.0 - rank,
                "elpd_diff": 0.0 if n == "model_a" else 1.0,
                "dse": 0.0 if n == "model_a" else 5.0,
                "dse_clustered": 0.0 if n == "model_a" else 5.0,
                "weight": 1.0 if n == "model_a" else 0.0,
                "loo_unreliable": False,
            }
            for rank, n in enumerate(model_zoo._manifest_names(models_dir))
        }

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "compare_table", fake_compare)
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    # The stub fit is not a real trace: pass the convergence gate.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, models_dir: object())


def _fake_spawn(candidate_dir, docs, **kwargs):
    (candidate_dir / "candidate.py").write_text("# new_idea\n", encoding="utf-8")
    (candidate_dir / "hypothesis.md").write_text("People use H.\n", encoding="utf-8")
    (candidate_dir / "model_name.txt").write_text("new_idea\n", encoding="utf-8")
    return True


def _run(tmp_path, monkeypatch, **loop_kwargs):
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", _fake_spawn)
    results_dir = tmp_path / "model_loop"
    run_pymc_inner_loop(
        responses,
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=1,
        enable_critique=False,
        ledger_context="experiment1",
        **loop_kwargs,
    )
    return results_dir


def test_loop_writes_its_novelty_pool_and_gates_every_candidate_on_it(
    tmp_path, monkeypatch
):
    _patch_scoring(monkeypatch)
    gate_calls = []

    def capturing_gate(name, models_dir, responses_path, *, pool_rows, **kwargs):
        gate_calls.append({"name": name, "pool_rows": list(pool_rows)})
        return None, float("inf")

    monkeypatch.setattr(model_zoo, "_min_prediction_rmse", capturing_gate)

    results_dir = _run(tmp_path, monkeypatch)

    pool_path = results_dir / NOVELTY_POOL_FILENAME
    assert pool_path.exists()
    recorded = json.loads(pool_path.read_text(encoding="utf-8"))
    assert recorded == novelty_pool_rows()
    assert [c["name"] for c in gate_calls] == ["new_idea"]
    assert gate_calls[0]["pool_rows"] == recorded


def test_disabled_gate_writes_no_pool(tmp_path, monkeypatch):
    _patch_scoring(monkeypatch)

    def tripwire(*a, **k):
        raise AssertionError("gate must not run when the threshold is 0")

    monkeypatch.setattr(model_zoo, "_min_prediction_rmse", tripwire)
    results_dir = _run(tmp_path, monkeypatch, novelty_rmse_threshold=0.0)
    assert not (results_dir / NOVELTY_POOL_FILENAME).exists()


# ── Every project seed model can bind the pool ────────────────────────


def test_every_project_seed_model_binds_the_novelty_pool():
    """The gate compares every admitted model on bare stimulus rows plus a
    dummy response, so every project seed — the models a run starts with —
    must bind the pool through its own feature hook. No MCMC: this only
    builds each model's ``pm.set_data`` dict for the pool."""
    from src.models.data_binding import make_stim_data as real_make_stim_data
    from src.models.model_loading import load_pymc_model as real_load_pymc_model
    from src.models.model_manifest import read_manifest_names
    from src.pipelines.outer_loop.orchestrator import project_seed_models_dir

    seed_dir = project_seed_models_dir("subjective_randomness")
    names = read_manifest_names(seed_dir)
    assert names, "the project has no seed models"
    rows = [{**row, "chose_left": 0} for row in novelty_pool_rows()]
    for name in names:
        model = real_load_pymc_model(name, seed_dir)
        stim_data = real_make_stim_data(model, rows)
        assert stim_data, name


# ── An undefined p_left on the pool: reject the candidate, never crash ──
#
# A model can break on pairs unlike any it was trained on (the recovery
# evaluation met one, commit 4de536c). The gate used to let the resulting
# InvalidPredictions escape, ending the cell; and one *admitted* model like
# that crashed every later candidate's admission.


class _UndefinedOn(_FakeFitted):
    """``p_left`` is ``predict(row)``, except NaN on the rows ``bad`` selects:
    ``predict_p_left`` raises ``InvalidPredictions`` as the real one does."""

    def __init__(self, predict, bad):
        super().__init__(predict)
        self._bad = bad

    def predict_p_left(self, stim_data, **kwargs):
        from src.models.pymc_inference import InvalidPredictions

        rows = stim_data["rows"]
        draws = np.tile([self._predict(row) for row in rows], (4, 1)).astype(float)
        draws[:, [i for i, row in enumerate(rows) if self._bad(row)]] = np.nan
        raise InvalidPredictions("values must be finite and in [0, 1].", draws)


def _long(row):
    return len(row["sequence_a"]) == 8


def test_a_candidate_undefined_on_the_pool_is_rejected_with_the_reason(
    tmp_path, monkeypatch, capsys
):
    models_dir = _models_dir(tmp_path, ["seed_a"])
    responses = _write_responses(tmp_path)
    _stub_admission_gates(monkeypatch)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "seed_a": _FakeFitted(lambda row: 0.5),
            "blows_up": _UndefinedOn(lambda row: 0.9, _long),
        },
    )
    ledger = model_zoo.HypothesisLedger.create(
        tmp_path / "ledger.jsonl", inherit_from=None
    )

    admission = model_zoo._admit_candidate_with_reason(
        _candidate(tmp_path), models_dir, "blows_up", responses, ledger=ledger
    )

    assert not admission.admitted
    n_long = sum(_long(row) for row in novelty_pool_rows())
    assert f"undefined (NaN or outside [0, 1]) on {n_long} of the" in admission.reason
    assert " vs " in admission.reason  # names example pairs
    assert not (models_dir / "blows_up.py").exists()
    (entry,) = ledger.entries()
    assert entry.outcome == "rejected" and "undefined" in entry.detail


def test_an_admitted_model_undefined_on_some_pool_stimuli_is_compared_on_the_rest(
    tmp_path, monkeypatch, capsys
):
    models_dir = _models_dir(tmp_path, ["partly_nan"])
    responses = _write_responses(tmp_path)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            # Agrees with the candidate wherever it is defined.
            "partly_nan": _UndefinedOn(lambda row: 0.7, _long),
            "candidate_x": _FakeFitted(lambda row: 0.7),
        },
    )

    nearest, rmse = _min_prediction_rmse(
        "candidate_x", models_dir, responses, pool_rows=novelty_pool_rows()
    )

    assert (nearest, rmse) == ("partly_nan", pytest.approx(0.0))
    assert "comparing on the rest" in capsys.readouterr().out


def test_an_admitted_model_undefined_everywhere_is_left_out_of_the_comparison(
    tmp_path, monkeypatch, capsys
):
    models_dir = _models_dir(tmp_path, ["all_nan", "seed_a"])
    responses = _write_responses(tmp_path)
    _stub_prediction_plumbing(
        monkeypatch,
        {
            "all_nan": _UndefinedOn(lambda row: 0.7, lambda row: True),
            "seed_a": _FakeFitted(lambda row: 0.5),
            "candidate_x": _FakeFitted(lambda row: 0.6),
        },
    )

    nearest, rmse = _min_prediction_rmse(
        "candidate_x", models_dir, responses, pool_rows=novelty_pool_rows()
    )

    assert (nearest, rmse) == ("seed_a", pytest.approx(0.1))
    assert "left out of the comparison" in capsys.readouterr().out
