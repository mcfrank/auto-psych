"""Tests for the LOO design-effect analysis (``src/subjective_randomness/
loo_design_effect.py`` and ``scripts/subjective_randomness/loo_design_effect.py``).

The inner loop prunes on ``elpd_diff > 2·dse`` from a trial-level PSIS-LOO,
which treats the participants' responses to one stimulus as independent
observations. The analysis re-reads a finished holdout cell's cached fits
(no new MCMC) and reports, for every archived prune decision, the same
comparison under two other units — a stimulus-clustered standard error of
the trial-level ELPD difference, and a leave-one-stimulus-out PSIS-LOO —
so the pruning unit can be chosen from a measured design effect.

The synthetic cell here is built so the answers are known in closed form:
every model's log-likelihood draws are identical across the participants of
one stimulus, so the clustered dse must be exactly ``sqrt(participants)``
times the trial-level dse, and the loser's margin is chosen to be pruned at
trial level but kept under the clustered unit — one flip.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.models import pymc_inference as pi

GT = "held_out_gt"
FIT_KWARGS = {"draws": 100, "tune": 50, "chains": 2, "target_accept": 0.8}
N_STIMULI = 4
N_PARTICIPANTS = 5
N_CHAINS, N_DRAWS = 2, 100


def _rows():
    """A responses.csv layout: every participant answers every stimulus."""
    rows = []
    trial = 0
    for participant in range(N_PARTICIPANTS):
        for stim in range(N_STIMULI):
            rows.append(
                {
                    "sequence_a": "HT" * (stim + 1),
                    "sequence_b": "TH" * (stim + 1),
                    "participant_id": participant,
                    "trial_index": trial,
                    "chose_left": trial % 2,
                }
            )
            trial += 1
    return rows


def _write_csv(path: Path, rows) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _log_likelihood(rng, stimulus_offsets):
    """``(chain, draw, trial)`` draws: per stimulus one noise pattern shared by
    all of its participants (so trials of a stimulus are perfectly correlated),
    shifted down by that stimulus's offset for the model in question."""
    rows = _rows()
    per_stimulus = -0.5 + rng.normal(0.0, 0.02, size=(N_CHAINS, N_DRAWS, N_STIMULI))
    arr = np.empty((N_CHAINS, N_DRAWS, len(rows)))
    for i, row in enumerate(rows):
        stim = len(row["sequence_a"]) // 2 - 1
        arr[:, :, i] = per_stimulus[:, :, stim] - stimulus_offsets[stim]
    return arr


def _idata(log_lik, rng):
    import arviz as az
    import xarray as xr

    coords = {"chain": np.arange(N_CHAINS), "draw": np.arange(N_DRAWS)}
    ll = xr.Dataset(
        {"response": (("chain", "draw", "response_dim_0"), log_lik)},
        coords={**coords, "response_dim_0": np.arange(log_lik.shape[-1])},
    )
    post = xr.Dataset(
        {"theta": (("chain", "draw"), rng.normal(size=(N_CHAINS, N_DRAWS)))},
        coords=coords,
    )
    return az.InferenceData(posterior=post, log_likelihood=ll)


# The three models of the cell: ``alpha`` and ``beta`` are the protected seeds
# (``alpha`` best), ``gamma`` the admitted candidate the loop pruned. gamma
# loses 1 nat per trial on the first stimulus only: over 5 participants that
# is a 5-nat total. Trial-level: the 20 pointwise differences are five 1s and
# fifteen 0s, dse = sqrt(20 · 0.1875) ≈ 1.94, so 5 > 2·dse — pruned. Clustered
# by stimulus: sums [5, 0, 0, 0], dse = sqrt(4 · 4.6875) ≈ 4.33 = sqrt(5)·1.94,
# so 5 < 2·dse — kept. beta trails alpha by 0.05 nats on every trial.
OFFSETS = {
    "alpha": np.zeros(N_STIMULI),
    "beta": np.full(N_STIMULI, 0.05),
    "gamma": np.array([1.0, 0.0, 0.0, 0.0]),
}


def _synthetic_fits():
    rng = np.random.default_rng(7)
    return {name: _idata(_log_likelihood(rng, offsets), rng) for name, offsets in OFFSETS.items()}


def _oracle_compare(fits, names):
    """arviz's own trial-level comparison, computed independently of the module
    under test, to write the ledger margins and the history ELPDs from."""
    import arviz as az

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loos = {name: az.loo(fits[name], pointwise=True) for name in names}
        return az.compare(loos, ic="loo"), loos


def build_cell(root: Path, *, archived: bool = True, carried_seed_step: bool = False) -> Path:
    """One holdout cell (``run1/<gt>/``) with one experiment: seed step over
    alpha+beta, then round 0 admits gamma and prunes it. The cached ``.nc``
    fits are named by the same fingerprint ``fit_model`` writes.

    With ``carried_seed_step`` the cell has two experiments instead: experiment
    1 is the seed step alone, and experiment 2 (same responses, so the same
    cached fits) opens with gamma carried into its seed step — where the loop
    never prunes, however far behind a model is — and prunes it at round 0."""
    cell = root / "run1" / GT
    staging = cell / "_staging"
    loop = staging / "_runs" / GT / "experiment1" / "model_loop"
    models_dir = loop / "models"
    pruned_dir = models_dir / "pruned"
    pruned_dir.mkdir(parents=True)
    responses = loop / "responses.csv"
    _write_csv(responses, _rows())
    for name in ("alpha", "beta"):
        (models_dir / f"{name}.py").write_text(f"# synthetic model {name}\n", encoding="utf-8")
    (pruned_dir / "gamma.py").write_text("# synthetic model gamma\n", encoding="utf-8")

    fits = _synthetic_fits()
    cache_dir = cell / "mcmc_cache"
    cache_dir.mkdir(parents=True)
    for name, idata in fits.items():
        source_dir = pruned_dir if name == "gamma" else models_dir
        settings = pi.resolve_fit_settings(name, source_dir, dict(FIT_KWARGS))
        fingerprint = pi.fit_fingerprint(name, source_dir, responses, settings)
        idata.to_netcdf(str(cache_dir / f"{name}.{fingerprint}.nc"))

    seed_cmp, seed_loos = _oracle_compare(fits, ["alpha", "beta"])
    round_cmp, round_loos = _oracle_compare(fits, ["alpha", "beta", "gamma"])
    gamma = round_cmp.loc["gamma"]
    history = [
        {
            "step": 0,
            "iteration": None,
            "best_model": "alpha",
            "argmax_model": "alpha",
            "excluded_unreliable": [],
            "posteriors": {"alpha": 1.0, "beta": 0.0},
            "elpd_loo": {n: round(float(seed_loos[n].elpd_loo), 4) for n in ("alpha", "beta")},
        },
        {
            "step": 1,
            "iteration": 0,
            "best_model": "alpha",
            "argmax_model": "alpha",
            "excluded_unreliable": [],
            "posteriors": {"alpha": 1.0, "beta": 0.0},
            "elpd_loo": {n: round(float(round_loos[n].elpd_loo), 4) for n in ("alpha", "beta")},
            "pruned": ["gamma"],
        },
    ]
    margin = (
        f"{float(gamma['elpd_diff']):.1f} nats behind alpha "
        f"({float(gamma['elpd_diff']) / float(gamma['dse']):.1f}× dse)"
    )
    if carried_seed_step:
        # Experiment 1: the seed step alone. Experiment 2: gamma is already
        # in the set at the seed step (all three scored, nothing pruned), then
        # round 0 prunes it. Same responses bytes, so the same cached fits.
        carried_seed = {**history[1], "step": 0, "iteration": None,
                        "elpd_loo": {**history[1]["elpd_loo"],
                                     "gamma": round(float(round_loos["gamma"].elpd_loo), 4)}}
        carried_seed.pop("pruned")
        (loop / "history.json").write_text(json.dumps([history[0]]), encoding="utf-8")
        (loop / "attempted_hypotheses.jsonl").write_text("", encoding="utf-8")
        loop2 = staging / "_runs" / GT / "experiment2" / "model_loop"
        shutil.copytree(loop, loop2)
        (loop2 / "history.json").write_text(json.dumps([carried_seed, history[1]]), encoding="utf-8")
        ledger_context = "experiment2 round 0"
        ledger_path = loop2 / "attempted_hypotheses.jsonl"
    else:
        (loop / "history.json").write_text(json.dumps(history), encoding="utf-8")
        ledger_context = "experiment1 round 0"
        ledger_path = loop / "attempted_hypotheses.jsonl"
    ledger = [
        {
            "name": "gamma",
            "outcome": "admitted",
            "detail": "",
            "hypothesis": "gamma mechanism",
            "context": f"{ledger_context} candidate 1",
        },
        {
            "name": "gamma",
            "outcome": "pruned",
            "detail": margin,
            "hypothesis": "gamma mechanism",
            "context": ledger_context,
        },
    ]
    ledger_path.write_text("".join(json.dumps(e) + "\n" for e in ledger), encoding="utf-8")
    (cell / "holdout.json").write_text(
        json.dumps({"fit_kwargs": FIT_KWARGS, "n_participants": N_PARTICIPANTS}),
        encoding="utf-8",
    )
    if archived:
        with tarfile.open(cell / "agent_runs.tar.gz", "w:gz") as tar:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(staging)))
        shutil.rmtree(staging)
    else:
        shutil.move(str(staging / "_runs"), str(cell / "repo" / "_runs"))
        shutil.rmtree(staging)
    return cell


# ─────────────────────────────────────────────
# Integration: one archived cell, both units, one flip
# ─────────────────────────────────────────────


def test_analyze_cell_reports_both_units_and_the_flip(tmp_path):
    from src.subjective_randomness.loo_design_effect import analyze_cell

    cell = build_cell(tmp_path / "sweep")
    record = analyze_cell(cell, tmp_path / "work")

    assert record["cell"] == f"run1/{GT}"
    assert record["protected"] == ["alpha", "beta"]
    [experiment] = record["experiments"]
    assert experiment["n_trials"] == N_STIMULI * N_PARTICIPANTS
    assert experiment["n_stimuli"] == N_STIMULI

    seed_step, round_step = experiment["steps"]
    assert sorted(r["name"] for r in seed_step["rows"]) == ["alpha", "beta"]
    assert sorted(r["name"] for r in round_step["rows"]) == ["alpha", "beta", "gamma"]
    assert round_step["archived_pruned"] == ["gamma"]

    gamma = next(r for r in round_step["rows"] if r["name"] == "gamma")
    assert gamma["protected"] is False
    # The trial-level numbers are the loop's own (the ledger margin agrees).
    assert gamma["elpd_diff_trial"] == pytest.approx(5.0, abs=0.3)
    assert gamma["dse_trial"] == pytest.approx(np.sqrt(20 * 0.1875), rel=0.1)
    assert gamma["prune_trial"] is True
    # Clustering by stimulus: identical draws within a stimulus give exactly
    # sqrt(participants) — the design effect of perfectly correlated trials.
    assert gamma["dse_cluster"] / gamma["dse_trial"] == pytest.approx(np.sqrt(N_PARTICIPANTS))
    assert gamma["ratio_cluster"] == pytest.approx(np.sqrt(N_PARTICIPANTS))
    assert gamma["prune_cluster"] is False
    # Leave-one-stimulus-out PSIS-LOO: its own diff, dse and reliability verdict.
    assert np.isfinite(gamma["elpd_diff_grouped"]) and gamma["elpd_diff_grouped"] > 0
    assert np.isfinite(gamma["dse_grouped"]) and gamma["dse_grouped"] > 0
    assert isinstance(gamma["unreliable_grouped"], bool)
    assert isinstance(gamma["prune_grouped"], bool)

    alpha = next(r for r in round_step["rows"] if r["name"] == "alpha")
    assert alpha["rank_trial"] == 0 and alpha["elpd_diff_trial"] == 0.0
    assert alpha["prune_trial"] is False and alpha["prune_cluster"] is False

    decisions = record["summary"]
    assert decisions["n_decisions"] == 1  # one non-protected model at one step
    assert decisions["n_pruned_archived"] == 1
    assert decisions["n_prune_trial"] == 1
    assert decisions["n_prune_cluster"] == 0
    assert decisions["n_flip_trial_vs_cluster"] == 1
    assert decisions["ratio_cluster"]["median"] == pytest.approx(np.sqrt(N_PARTICIPANTS))


def test_a_carried_model_behind_at_a_seed_step_is_reported_but_is_no_decision(tmp_path):
    """The loop prunes only after a candidate round: a model carried into the
    next experiment can be far behind at that experiment's seed step and still
    be scored there. The replication reports the seed-step row (the rule would
    prune it) without treating the step as a prune decision or as a
    disagreement with the record."""
    from src.subjective_randomness.loo_design_effect import analyze_cell

    cell = build_cell(tmp_path / "sweep", carried_seed_step=True)
    record = analyze_cell(cell, tmp_path / "work")

    first, second = record["experiments"]
    assert [s["archived_pruned"] for s in first["steps"]] == [[]]
    seed_step, round_step = second["steps"]
    assert seed_step["iteration"] is None and seed_step["archived_pruned"] == []
    gamma_at_seed = next(r for r in seed_step["rows"] if r["name"] == "gamma")
    assert gamma_at_seed["prune_trial"] is True  # what the rule would do
    assert round_step["archived_pruned"] == ["gamma"]
    assert record["summary"]["n_decisions"] == 1  # round 0 of experiment 2 only
    assert record["summary"]["n_pruned_archived"] == 1


def test_analyze_cell_reads_a_kept_repo_copy(tmp_path):
    from src.subjective_randomness.loo_design_effect import analyze_cell

    cell = build_cell(tmp_path / "sweep", archived=False)
    record = analyze_cell(cell, tmp_path / "work")
    assert record["summary"]["n_flip_trial_vs_cluster"] == 1


def test_analyze_cell_fails_loudly_when_the_cache_lacks_a_fit(tmp_path):
    from src.subjective_randomness.loo_design_effect import analyze_cell

    cell = build_cell(tmp_path / "sweep")
    for path in (cell / "mcmc_cache").glob("gamma.*.nc"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="gamma"):
        analyze_cell(cell, tmp_path / "work")


def test_analyze_cell_fails_loudly_when_history_disagrees_with_the_cache(tmp_path):
    """A recomputed ELPD that does not match the recorded one means the cached
    fit is not the fit the loop scored: the mapping is wrong, not the data."""
    from src.subjective_randomness.loo_design_effect import analyze_cell

    cell = build_cell(tmp_path / "sweep")
    tar_path = cell / "agent_runs.tar.gz"
    staging = tmp_path / "restage"
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(staging)
    history_path = staging / "_runs" / GT / "experiment1" / "model_loop" / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history[1]["elpd_loo"]["alpha"] -= 3.0
    history_path.write_text(json.dumps(history), encoding="utf-8")
    tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(staging)))
    with pytest.raises(ValueError, match="alpha"):
        analyze_cell(cell, tmp_path / "work")


# ─────────────────────────────────────────────
# Units
# ─────────────────────────────────────────────


def test_fit_fingerprint_is_what_fit_model_names_the_cache_file(tmp_path):
    """One public function owns the ``.nc`` name, so an offline reader of a
    cache can locate a fit without re-deriving the formula."""
    import hashlib

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.py").write_text("# model\n", encoding="utf-8")
    responses = tmp_path / "responses.csv"
    responses.write_text("sequence_a,sequence_b,participant_id,trial_index,chose_left\nHT,TH,0,0,1\n")
    settings = pi.resolve_fit_settings("m", models_dir, {"draws": 10})

    fingerprint = pi.fit_fingerprint("m", models_dir, responses, settings)

    expected = hashlib.sha256(
        (
            pi._sha256_file(models_dir / "m.py")
            + pi._sha256_file(responses)
            + pi._sampler_signature(settings)
        ).encode("utf-8")
    ).hexdigest()[:16]
    assert fingerprint == expected
    assert pi.cached_fit_path(tmp_path / "cache", "m", fingerprint) == (
        tmp_path / "cache" / f"m.{fingerprint}.nc"
    )
    responses.write_text(responses.read_text() + "TH,HT,0,1,0\n")
    assert pi.fit_fingerprint("m", models_dir, responses, settings) != fingerprint


def test_stimulus_groups_share_an_id_per_ordered_pair():
    from src.subjective_randomness.loo_design_effect import stimulus_groups

    rows = [
        {"sequence_a": "HT", "sequence_b": "TH", "participant_id": 0},
        {"sequence_a": "HH", "sequence_b": "TT", "participant_id": 0},
        {"sequence_a": "HT", "sequence_b": "TH", "participant_id": 1},
        {"sequence_a": "TH", "sequence_b": "HT", "participant_id": 1},  # mirrored: a different presentation
    ]
    groups = stimulus_groups(rows)
    assert groups.tolist() == [0, 1, 0, 2]


def test_stimulus_groups_requires_the_stimulus_columns():
    from src.subjective_randomness.loo_design_effect import stimulus_groups

    with pytest.raises(KeyError, match="sequence_b"):
        stimulus_groups([{"sequence_a": "HT", "participant_id": 0}])


def test_grouped_log_likelihood_sums_each_stimulus_per_draw():
    from src.subjective_randomness.loo_design_effect import grouped_log_likelihood

    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    groups = np.array([0, 1, 0, 1])
    grouped = grouped_log_likelihood(arr, groups)
    assert grouped.shape == (2, 3, 2)
    np.testing.assert_allclose(grouped[..., 0], arr[..., 0] + arr[..., 2])
    np.testing.assert_allclose(grouped[..., 1], arr[..., 1] + arr[..., 3])


def test_elpd_difference_matches_arviz_compare():
    from src.subjective_randomness.loo_design_effect import elpd_difference

    fits = _synthetic_fits()
    cmp, loos = _oracle_compare(fits, ["alpha", "gamma"])
    diff, dse = elpd_difference(
        np.asarray(loos["alpha"].loo_i), np.asarray(loos["gamma"].loo_i)
    )
    assert diff == pytest.approx(float(cmp.loc["gamma", "elpd_diff"]))
    assert dse == pytest.approx(float(cmp.loc["gamma", "dse"]))


def test_cluster_dse_is_sqrt_group_size_for_perfectly_correlated_trials():
    from src.subjective_randomness.loo_design_effect import cluster_dse, elpd_difference

    best = np.repeat([-1.0, -2.0, -1.5, -0.5], 5)
    other = best - np.repeat([1.0, 0.0, 0.0, 0.0], 5)
    groups = np.repeat(np.arange(4), 5)
    _, dse_trial = elpd_difference(best, other)
    assert cluster_dse(best, other, groups) == pytest.approx(np.sqrt(5) * dse_trial)
    # Singleton groups are the trial-level unit.
    assert cluster_dse(best, other, np.arange(20)) == pytest.approx(dse_trial)


def test_comparison_models_are_the_survivors_plus_the_pruned():
    from src.subjective_randomness.loo_design_effect import comparison_models

    step = {"elpd_loo": {"b": -1.0, "a": -2.0}, "pruned": ["c"]}
    assert comparison_models(step) == ["a", "b", "c"]
    assert comparison_models({"elpd_loo": {"a": -1.0}}) == ["a"]


def test_prune_rule_replicates_the_loop():
    """Non-protected, PSIS-LOO-reliable, ``elpd_diff > 2·dse`` — and nothing at
    all when the rank-0 baseline is unreliable (see ``_prune_losers``)."""
    from src.subjective_randomness.loo_design_effect import prune_set

    table = {
        "best": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "unreliable": False},
        "seed": {"rank": 1, "elpd_diff": 50.0, "dse": 5.0, "unreliable": False},
        "loser": {"rank": 2, "elpd_diff": 30.0, "dse": 5.0, "unreliable": False},
        "close": {"rank": 3, "elpd_diff": 9.0, "dse": 5.0, "unreliable": False},
        "shaky": {"rank": 4, "elpd_diff": 90.0, "dse": 5.0, "unreliable": True},
    }
    assert prune_set(table, protected={"best", "seed"}) == ["loser"]
    unreliable_baseline = {**table, "best": {**table["best"], "unreliable": True}}
    assert prune_set(unreliable_baseline, protected={"best", "seed"}) == []


# ─────────────────────────────────────────────
# The CLI: per-cell JSON and the sweep table
# ─────────────────────────────────────────────


def test_cli_writes_per_cell_json_and_a_sweep_table(tmp_path):
    from scripts.subjective_randomness.loo_design_effect import Args, main

    sweep = tmp_path / "sweep"
    build_cell(sweep)
    out_dir = tmp_path / "out"
    main(Args(sweep=sweep, out_dir=out_dir, work_dir=tmp_path / "work"))

    record = json.loads((out_dir / f"run1__{GT}.json").read_text(encoding="utf-8"))
    assert record["summary"]["n_flip_trial_vs_cluster"] == 1
    table = (out_dir / "loo_design_effect.md").read_text(encoding="utf-8")
    assert f"run1/{GT}" in table
    assert "1 / 1" in table  # pruned by the loop / decisions
    assert "Totals" in table


def test_cli_report_only_rebuilds_the_table_from_the_json_records(tmp_path):
    from scripts.subjective_randomness.loo_design_effect import Args, main

    sweep = tmp_path / "sweep"
    build_cell(sweep)
    out_dir = tmp_path / "out"
    main(Args(sweep=sweep, out_dir=out_dir, work_dir=tmp_path / "work"))
    (out_dir / "loo_design_effect.md").unlink()

    main(Args(sweep=sweep, out_dir=out_dir, work_dir=tmp_path / "work", report_only=True))
    assert (out_dir / "loo_design_effect.md").exists()


def test_cli_selects_cells_and_fails_on_an_unknown_label(tmp_path):
    from scripts.subjective_randomness.loo_design_effect import Args, main

    sweep = tmp_path / "sweep"
    build_cell(sweep)
    with pytest.raises(FileNotFoundError, match="run9/nowhere"):
        main(Args(sweep=sweep, out_dir=tmp_path / "out", work_dir=tmp_path / "work", cells="run9/nowhere"))
    with pytest.raises(FileNotFoundError):
        main(Args(sweep=tmp_path / "missing", out_dir=tmp_path / "out", work_dir=tmp_path / "work"))


def _row(name, *, rank_trial, rank_grouped, ratio_cluster, ratio_grouped, protected=False):
    return {
        "name": name, "protected": protected, "rank_trial": rank_trial,
        "rank_grouped": rank_grouped, "ratio_cluster": ratio_cluster,
        "ratio_grouped": ratio_grouped, "prune_trial": False, "prune_cluster": False,
        "prune_grouped": False, "prune_grouped_ignoring_reliability": False,
        "unreliable_grouped": False,
    }


def test_summary_ratio_excludes_the_best_model_of_each_unit():
    """A unit's best model has dse 0 by construction, so its ratio is not a
    design effect. When the leave-one-stimulus-out best differs from the
    trial-level best (a statistical tie), that row's grouped ratio is
    undefined and must not enter the distribution as a 0."""
    from src.subjective_randomness.loo_design_effect import summarise_cell

    experiments = [{
        "steps": [{
            "iteration": 0,
            "archived_pruned": [],
            "rows": [
                _row("a", rank_trial=0, rank_grouped=1, ratio_cluster=None, ratio_grouped=1.5),
                _row("b", rank_trial=1, rank_grouped=0, ratio_cluster=2.0, ratio_grouped=0.0),
                _row("c", rank_trial=2, rank_grouped=2, ratio_cluster=3.0, ratio_grouped=2.5),
            ],
        }],
    }]
    summary = summarise_cell(experiments)
    assert summary["ratio_cluster"]["n"] == 2 and summary["ratio_cluster"]["min"] == 2.0
    assert summary["ratio_grouped"]["n"] == 1 and summary["ratio_grouped"]["median"] == 2.5


def test_a_near_constant_stimulus_group_is_snapped_so_its_loo_term_is_exact():
    """Forty clipped trials summed per draw leave floating-point noise (~1e-14)
    on a group whose log-likelihood is really constant. arviz gives an exactly
    constant column a finite, exact LOO term but returned NaN weights for such
    a near-constant one on a real fit (run1/local_representativeness,
    misweighted_bayesian_markov: spread 4e-14 over 8000 draws; the failure is
    inside arviz's tail fit and is not reproduced synthetically here), so
    groups within the exact-trial tolerance are snapped to their mean and
    PSIS-LOO on the snapped array carries the exact term."""
    import arviz as az
    import xarray as xr

    from src.models.loo_reliability import EXACT_TRIAL_LOGLIK_SPREAD
    from src.subjective_randomness.loo_design_effect import snap_exact_groups

    rng = np.random.default_rng(3)
    grouped = -10.0 + rng.normal(0.0, 0.05, size=(2, 200, 5))
    grouped[:, :, 2] = -27.7 + rng.uniform(-2e-14, 2e-14, size=(2, 200))
    snapped = snap_exact_groups(grouped)
    assert np.ptp(snapped[:, :, 2]) == 0.0
    assert snapped[:, :, 2].mean() == pytest.approx(grouped[:, :, 2].mean())
    others = [0, 1, 3, 4]
    np.testing.assert_array_equal(snapped[:, :, others], grouped[:, :, others])
    # A column with genuine spread above the tolerance is untouched.
    assert np.ptp(grouped[:, :, 0]) > EXACT_TRIAL_LOGLIK_SPREAD

    coords = {"chain": np.arange(2), "draw": np.arange(200), "stimulus": np.arange(5)}
    post = xr.Dataset({"theta": (("chain", "draw"), rng.normal(size=(2, 200)))}, coords=coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = az.loo(az.InferenceData(posterior=post, log_likelihood=xr.Dataset(
            {"response": (("chain", "draw", "stimulus"), snapped)}, coords=coords)), pointwise=True)
    assert np.isfinite(float(fixed.elpd_loo))
    assert float(fixed.loo_i[2]) == pytest.approx(-27.7, abs=1e-9)
