"""The subjective-randomness data behind the paper's Figures 2 and 3
(``scripts/paper/sr_figure_data.py``).

Per run (Figure 2), the final experiment's winner and the run's starting
models (seeds) are fitted on the run's final data and compared by PSIS-LOO.
For Figure 3 every run's winner and the seeds are fitted on all runs' data
together; the best winner and the best seed there supply per-condition
(unordered pair) predictions beside the observed choice proportions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.paths import PYMC_MODEL_FIXTURES_DIR

SEEDS = ["seed_wide", "seed_lapse"]

_REPRESENTATIVENESS = (PYMC_MODEL_FIXTURES_DIR / "representativeness.py").read_text()
_SIGMOID = 'pm.math.sigmoid(tau * (dev_b - dev_a))'
# Variants of the representativeness fixture, whose PSIS-LOO is reliable on
# the fixture data (the fair-coin fixture's is not: most trials have high k).
MODEL_SOURCES = {
    "seed_wide": _REPRESENTATIVENESS,
    "seed_lapse": _REPRESENTATIVENESS.replace(_SIGMOID, f"0.1 + 0.8 * {_SIGMOID}"),
    "narrow": _REPRESENTATIVENESS.replace("sigma=5.0", "sigma=1.0"),
    "small_lapse": _REPRESENTATIVENESS.replace(_SIGMOID, f"0.02 + 0.96 * {_SIGMOID}"),
}
for _name, _source in MODEL_SOURCES.items():
    assert _name == "seed_wide" or _source != _REPRESENTATIVENESS, _name


REPEATS = 10  # 30 fixture trials are too few for a reliable PSIS-LOO
REMOVED = "gone"  # the model each fixture run pruned in experiment 1 (refitted)
LATE = "gone_late"  # pruned at the end of experiment 2: its ELPD is the loop's
LATE_ELPD = -170.0


def _fixture_rows():
    with open(PYMC_MODEL_FIXTURES_DIR / "responses.csv", newline="") as f:
        return list(csv.DictReader(f))


def _write_responses(path: Path) -> None:
    """The fixture's trials repeated, each repeat as new participants."""
    rows = _fixture_rows()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for k in range(REPEATS):
            for row in rows:
                writer.writerow({**row, "participant_id": f"{row['participant_id']}_{k}"})


def _write_run(run_root: Path, *, winner_source: str, winner: str) -> None:
    """A finished run in the collected layout: seeds and one removed model
    (``REMOVED``) pruned in experiment 1, experiment 2 final with its winner,
    pooled responses, the ledger and the export record."""
    sr = run_root / "subjective_randomness"
    pruned = sr / "experiment1" / "model_loop" / "models" / "pruned"
    pruned.mkdir(parents=True)
    for seed in SEEDS:
        (pruned / f"{seed}.py").write_text(MODEL_SOURCES[seed])
    (pruned / f"{REMOVED}.py").write_text(MODEL_SOURCES["narrow"])
    (sr / "starting_models.json").write_text(
        json.dumps({"starting_models": SEEDS, "starting_models_prunable": True})
    )
    (sr / "experiment1" / "model_loop" / "export_complete.json").write_text("{}")
    loop = sr / "experiment2" / "model_loop"
    (loop / "models").mkdir(parents=True)
    (loop / "models" / f"{winner}.py").write_text(MODEL_SOURCES[winner_source])
    _write_responses(loop / "responses.csv")
    # The loop's own ELPD-LOO of every model live at the end, on these data.
    (loop / "model_posterior.json").write_text(json.dumps({
        "best_model": winner, "n_trials": REPEATS * len(_fixture_rows()),
        "elpd_loo": {winner: -150.0, "rival": -155.0}, "excluded_unreliable": [],
    }))
    (loop / "export_complete.json").write_text("{}")
    (loop / "models" / "pruned").mkdir()
    (loop / "models" / "pruned" / f"{LATE}.py").write_text(MODEL_SOURCES["small_lapse"])
    (loop / "history.json").write_text(json.dumps([
        {"elpd_loo": {winner: -150.0, "rival": -155.0, LATE: LATE_ELPD}}]))
    ledger = [(winner, "admitted", "experiment1 round 1"), ("rival", "admitted", "experiment1 round 1"),
              (REMOVED, "admitted", "experiment1 round 1"),
              (REMOVED, "pruned", "experiment1 end of experiment"),
              (LATE, "admitted", "experiment2 round 1"),
              (LATE, "pruned", "experiment2 end of experiment")]
    (loop / "attempted_hypotheses.jsonl").write_text("".join(
        json.dumps({"name": n, "outcome": o, "detail": "20 nats behind w (3× dse)",
                    "hypothesis": f"{n} hypothesis", "context": c}) + "\n"
        for n, o, c in ledger))


def _run_main(tmp_path, **kwargs):
    from scripts.paper.sr_figure_data import Args, main

    runs_root = tmp_path / "collected"
    if not runs_root.exists():  # a rerun reads the runs the first call wrote
        _write_run(runs_root / "run1", winner_source="narrow", winner="winner_one")
        _write_run(runs_root / "run2", winner_source="small_lapse", winner="winner_two")
    out = tmp_path / "sr.json"
    settings = dict(runs_root=runs_root, runs=["run1", "run2"], out=out,
                    work_dir=tmp_path / "work", draws=500, tune=500, chains=2, pooled_draws=500,
                    pooled_accepted_fit_problems={}, **kwargs)
    main(Args(**settings, removed_shard=0, removed_shards=1))  # the removed models first
    main(Args(**settings))
    return json.loads(out.read_text())


@pytest.mark.slow
def test_discarding_each_runs_fits_leaves_no_fit_and_the_same_figure_data(tmp_path):
    kept = _run_main(tmp_path / "kept")
    discarded = _run_main(tmp_path / "discarded", discard_fits_after_run=True)

    assert not list((tmp_path / "discarded" / "work").glob("*/fit_cache/*.nc"))
    assert list((tmp_path / "kept" / "work").glob("*/fit_cache/*.nc"))
    assert list((tmp_path / "kept" / "work").glob("pooled/fit_cache/*.nc"))
    # Fits are seeded deterministically, so both give the same numbers.
    assert discarded["pooled"] == kept["pooled"]
    assert [c["pair"] for c in discarded["conditions"]] == [c["pair"] for c in kept["conditions"]]
    for a, b in zip(discarded["conditions"], kept["conditions"]):
        for key in ("n", "observed", "best_model", "best_seed"):
            assert a[key] == pytest.approx(b[key]), key
    for a, b in zip(discarded["runs"], kept["runs"]):
        assert a["vs_best_seed"] == b["vs_best_seed"]


@pytest.mark.slow
def test_figure_data_compares_each_winner_with_the_seeds_and_tabulates_all_runs_pooled(tmp_path):
    from scripts.paper.sr_figure_data import Args, main

    runs_root = tmp_path / "collected"
    _write_run(runs_root / "run1", winner_source="narrow", winner="winner_one")
    _write_run(runs_root / "run2", winner_source="small_lapse", winner="winner_two")
    out = tmp_path / "sr.json"
    settings = dict(runs_root=runs_root, runs=["run1", "run2"], out=out,
                    work_dir=tmp_path / "work", draws=500, tune=500, chains=2, pooled_draws=500,
                    pooled_tune=600, pooled_target_accept=0.9, pooled_random_seed=7,
                    pooled_accepted_fit_problems={})

    with pytest.raises(RuntimeError, match=r"no current scored record .*\['gone'\]"):
        main(Args(**settings))  # the removed models are scored first
    main(Args(**settings, removed_shard=0, removed_shards=1))
    main(Args(**settings))

    data = json.loads(out.read_text())
    assert data["domain"] == "subjective_randomness"
    assert [r["run"] for r in data["runs"]] == ["run1", "run2"]
    for run in data["runs"]:
        assert run["final_experiment"] == "experiment2"
        assert set(run["vs_best_seed"]) == {run["best_model"], *SEEDS}
        assert run["best_seed"] in SEEDS
        assert run["vs_best_seed"][run["best_seed"]] == {"diff": 0.0, "se": 0.0}
        for row in run["vs_best_seed"].values():
            assert row["se"] >= 0
        # The run's other models live at its end, from the loop's own fits,
        # against the best seed fitted here.
        assert run["others_at_end"] == {
            "rival": pytest.approx(-155.0 - run["elpd"][run["best_seed"]])}
        # The removed model, fitted here on the same data, against the best seed.
        assert run["pruned_outcome"] == {REMOVED: "pruned", LATE: "pruned"}
        assert run["pruned_source"] == {REMOVED: "refit", LATE: "loop"}
        assert run["pruned_unscored"] == {}
        best_seed = run["elpd"][run["best_seed"]]
        assert run["pruned_during_run"] == {
            REMOVED: pytest.approx(run["pruned_elpd"][REMOVED] - best_seed),
            LATE: pytest.approx(LATE_ELPD - best_seed)}
    assert "best_run" not in data

    # Figure 3: both runs' winners and the seeds fitted on both runs' data,
    # the runs' participants kept apart (both runs number theirs alike).
    pooled = data["pooled"]
    n_trials = REPEATS * len(_fixture_rows())
    n_people = len({r["participant_id"] for r in _fixture_rows()}) * REPEATS
    assert pooled["runs"] == ["run1", "run2"]
    assert pooled["n_trials"] == 2 * n_trials
    assert pooled["n_participants"] == 2 * n_people
    assert set(pooled["elpd"]) == {"winner_one", "winner_two", *SEEDS}
    assert pooled["winner_runs"] == {"winner_one": "run1", "winner_two": "run2"}
    # The pooled fits have sampler settings of their own, all models alike.
    assert pooled["fit_settings"] == {
        "draws": 500, "tune": 600, "chains": 2, "target_accept": 0.9, "random_seed": 7,
        "method": "nuts_no_jitter"}
    assert pooled["accepted_fit_problems"] == {}
    winners = ["winner_one", "winner_two"]
    assert pooled["best_model"] == max(winners, key=pooled["elpd"].get)
    assert pooled["best_seed"] == max(SEEDS, key=pooled["elpd"].get)

    conditions = data["conditions"]
    assert sum(c["n"] for c in conditions) == 2 * n_trials
    for c in conditions:
        assert c["pair"] == sorted(c["pair"])
        for key in ("observed", "best_model", "best_seed"):
            assert 0.0 <= c[key] <= 1.0


@pytest.mark.slow
def test_a_rerun_reuses_each_scored_run_and_refits_nothing(tmp_path, monkeypatch):
    first = _run_main(tmp_path, discard_fits_after_run=True)
    assert sorted(p.parent.name for p in (tmp_path / "work").glob("*/scored.json")) == [
        "pooled", "run1", "run2"]

    import src.models.pymc_inference as inference

    def no_fit(*args, **kwargs):
        raise AssertionError("a scored run was refitted")

    monkeypatch.setattr(inference, "fit_models_cached", no_fit)
    monkeypatch.setattr(inference, "fit_model", no_fit)
    second = _run_main(tmp_path)
    assert second == first


@pytest.mark.slow
def test_a_scored_run_whose_inputs_changed_is_scored_again(tmp_path):
    _run_main(tmp_path, discard_fits_after_run=True)
    record = tmp_path / "work" / "run1" / "scored.json"
    stale = json.loads(record.read_text())
    stale["inputs"]["fit_settings"]["draws"] = 1
    stale["record"]["best_model"] = "stale"
    record.write_text(json.dumps(stale))

    data = _run_main(tmp_path, discard_fits_after_run=True)
    assert data["runs"][0]["best_model"] == "winner_one"
