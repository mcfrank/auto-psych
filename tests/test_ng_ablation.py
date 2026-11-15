"""Ablating the number-game November final: the switchable rewrite, its variants, the data, the elimination."""

import json

import numpy as np
import pytest
from pyprojroot import here

from src.number_game.ablation import variants as V
from src.number_game.ablation import data as ng_data

# A real run's responses, for the checks that need some data but not the ablation's.
RUN1 = ng_data.run_responses("run1")

FINAL_DIR = here() / "data/results/number_game_dataset-nov26/collected/run2/number_game/experiment3/model_loop/models"
MECHANISMS = V.spec("ng").MECHANISMS
CUES = V.spec("ng").CUES


def bound(model):
    import pymc as pm

    from src.models.data_binding import extract_observed

    with model:
        pm.set_data(extract_observed(RUN1, model))
    return model


def variant_model(on, out_dir):
    from src.models.model_loading import load_pymc_model

    name, _ = V.write_variant("ng", on, out_dir)
    return bound(load_pymc_model(name, out_dir))


def jittered(model, seed):
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, 0.5, size=np.shape(v)) for k, v in model.initial_point().items()}


def p_yes(model, point):
    # p_yes is a function of the random variables: on their values, not on fresh draws.
    out = model.replace_rvs_by_values([model["p_yes"]])[0]
    fn = model.compile_fn(out, inputs=model.value_vars, on_unused_input="ignore")
    return np.asarray(fn({k: point[k] for k in (v.name for v in model.value_vars)}))


def constrained(model, point):
    """The free variables' values (constrained, by variable name) at an unconstrained point."""
    outs = model.replace_rvs_by_values(model.free_RVs)
    fn = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore")
    values = fn({v.name: point[v.name] for v in model.value_vars})
    return {rv.name: np.asarray(v, dtype="float64") for rv, v in zip(model.free_RVs, values)}


def p_yes_at(model, values):
    """p_yes with the free variables at these constrained values (boundary values allowed)."""
    import pytensor

    rvs = model.free_RVs
    stand_ins = [rv.type() for rv in rvs]
    out = pytensor.clone_replace(model["p_yes"], dict(zip(rvs, stand_ins)))
    fn = pytensor.function(stand_ins, out, on_unused_input="ignore")
    return np.asarray(fn(*[np.asarray(values[rv.name], dtype=rv.dtype) for rv in rvs]))


def cues_of(on):
    return [c for c in CUES if c in on]


def at_neutral(with_on, values, mechanism):
    """``values`` of the model with ``with_on`` with ``mechanism`` at its neutral value: a cue's own
    component of ``salience`` at 0, any other mechanism's parameters at ``NEUTRAL``."""
    out = dict(values)
    if mechanism in CUES:
        salience = values["salience"].copy()
        salience[cues_of(with_on).index(mechanism)] = 0.0
        out["salience"] = salience
    else:
        out.update({k: np.float64(v) for k, v in V.spec("ng").NEUTRAL[mechanism].items()})
    return out


def restricted(with_on, without_on, values):
    """``values`` for the model with ``without_on``: its own parameters, and the remaining cues'
    saliences renormalized (the judgment is a ratio of salience-weighted sums: scale-free)."""
    out = {k: v for k, v in values.items() if k != "salience"}
    keep = [cues_of(with_on).index(c) for c in cues_of(without_on)]
    salience = values["salience"][keep]
    out["salience"] = salience / salience.sum()
    return out


def without(m):
    """The largest valid set without ``m`` (every mechanism that needs it removed too)."""
    requires = V.spec("ng").REQUIRES
    on = set(MECHANISMS) - {m}
    while unmet := {x for x in on if x in requires and not on & set(requires[x])}:
        on -= unmet
    return on


def test_with_every_mechanism_on_the_variant_is_the_final(tmp_path):
    from src.models.model_loading import load_pymc_model

    original = bound(load_pymc_model(V.spec("ng").FINAL, FINAL_DIR))
    mine = variant_model(MECHANISMS, tmp_path)
    assert V.variant_name("ng", MECHANISMS) == "ng_full"
    assert {(v.name, tuple(original.initial_point()[v.name].shape)) for v in original.value_vars} == \
        {(v.name, tuple(mine.initial_point()[v.name].shape)) for v in mine.value_vars}
    logp_o, logp_m = original.compile_logp(), mine.compile_logp()
    for seed in range(4):
        point = jittered(original, seed)
        assert logp_m(point) == pytest.approx(logp_o(point), rel=1e-10, abs=1e-8)
        assert np.allclose(p_yes(mine, point), p_yes(original, point), rtol=0, atol=1e-12)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_each_mechanism_at_its_neutral_value_is_the_model_without_it(mechanism, tmp_path):
    off = without(mechanism)
    with_on = off | {mechanism}
    assert V.invalid_reason("ng", off) is None and V.invalid_reason("ng", with_on) is None
    with_m = variant_model(with_on, tmp_path)
    without_m = variant_model(off, tmp_path)
    # removing a mechanism removes parameters (or a cue's salience component) only
    assert {rv.name for rv in without_m.free_RVs} <= {rv.name for rv in with_m.free_RVs}
    values = constrained(with_m, jittered(with_m, 0))
    neutral = at_neutral(with_on, values, mechanism)
    assert np.allclose(p_yes_at(with_m, neutral), p_yes_at(without_m, restricted(with_on, off, neutral)),
                       rtol=0, atol=1e-10)
    # ... and away from it, the mechanism moves some prediction.
    moved = p_yes_at(with_m, values) - p_yes_at(without_m, restricted(with_on, off, values))
    assert np.abs(moved).max() > 1e-4


def test_a_modifier_goes_before_what_it_modifies_and_a_range_cue_stays():
    first = V.removals("ng", MECHANISMS)
    assert {"retain", "recency", "primacy", "doubt_rich", "doubt_rich_multi", "parity_keep", "doubt_band",
            "memory_limit", "range_low", "range_high", "parity", "span"} <= set(first)
    assert not {"last_digit", "doubt_single", "doubt_multi"} & set(first)
    assert "last_digit" in V.removals("ng", set(MECHANISMS) - {"retain", "recency", "primacy"})
    # The last-digit modifiers act through the last-digit cue's salience.
    assert V.invalid_reason("ng", ["range_low", "recency"]) == "recency needs one of ['last_digit']"
    # The ambiguity doubt's parity constraint needs some doubt; its magnitude band, multi-example doubt.
    assert V.invalid_reason("ng", ["range_low", "parity_keep"]) == \
        "parity_keep needs one of ['doubt_single', 'doubt_multi']"
    assert V.invalid_reason("ng", ["range_low", "doubt_single", "doubt_band"]) == \
        "doubt_band needs one of ['doubt_multi']"
    # A range cue stays: it keeps the judgment's denominator positive on every item.
    assert V.removals("ng", ["range_low", "range_high"]) == ["range_low", "range_high"]
    assert V.removals("ng", ["range_high"]) == []
    assert V.invalid_reason("ng", ["parity", "seed"]) == "one of ['range_low', 'range_high'] must stay"
    with pytest.raises(ValueError, match="unknown mechanisms"):
        V.canonical("ng", ["range_low", "telepathy"])


def test_variant_files_are_self_contained_and_named_by_what_was_removed(tmp_path):
    name, path = V.write_variant("ng", set(MECHANISMS) - {"memory_limit", "primacy"}, tmp_path)
    assert name == "ng_minus_prim_mem"
    assert "def build(on):" in path.read_text() and path.read_text().endswith("model = build(MECHANISMS_ON)\n")
    assert "# --- this variant (src.number_game.ablation.variants) ---" in path.read_text()
    with pytest.raises(ValueError, match="is not a model"):
        V.write_variant("ng", set(MECHANISMS) - {"range_low", "range_high"}, tmp_path)


def test_the_full_variant_has_the_finals_parameter_count():
    from src.models.model_loading import load_pymc_model

    original = load_pymc_model(V.spec("ng").FINAL, FINAL_DIR)
    point = original.initial_point()
    free = sum(int(np.prod(point[v.name].shape)) for v in original.value_vars)
    # 15 scalars and the 13 cues' saliences (12 free on the simplex)
    assert free == 27
    assert V.size("ng", MECHANISMS) == {"population": 27, "person_vectors": 0}


def test_the_full_and_the_smallest_variant_meet_the_data_contract(tmp_path):
    from src.models.model_contract import model_contract_violation

    smallest = set(MECHANISMS)
    while removable := V.removals("ng", smallest):
        smallest -= {removable[0]}
    assert smallest == {"range_high"}
    pooled = tmp_path / "responses.csv"
    ng_data.main(ng_data.Args(out=pooled))
    for on in (MECHANISMS, smallest):
        name, _ = V.write_variant("ng", on, tmp_path)
        assert model_contract_violation(name, tmp_path, pooled) is None, name


# --- the data and the CV units -----------------------------------------------------------


def test_the_data_are_all_three_runs_responses_pooled(tmp_path):
    import csv

    from scripts.paper.number_game_figure_data import pool_responses

    out = tmp_path / "responses.csv"
    ng_data.main(ng_data.Args(out=out))
    lines = out.read_text().splitlines()
    assert lines[0] == "number_set,target,trial_index,rating" and len(lines) - 1 == 22680
    # every run's rows, in run order, unchanged
    expected = []
    for run in ("run1", "run2", "run3"):
        with ng_data.run_responses(run).open(newline="") as f:
            expected += [tuple(r.values()) for r in csv.DictReader(f)]
    with out.open(newline="") as f:
        assert [tuple(r.values()) for r in csv.DictReader(f)] == expected
    # the same file as Figures 2 and 3's pooled fits
    figure = tmp_path / "figure_pooled.csv"
    pool_responses({run: ng_data.run_responses(run) for run in ("run1", "run2", "run3")}, figure)
    assert out.read_text() == figure.read_text()
    ng_data.main(ng_data.Args(out=out))  # up to date: kept
    out.write_text(RUN1.read_text())
    with pytest.raises(ValueError, match="is not"):
        ng_data.main(ng_data.Args(out=out))


def test_folds_and_clusters_are_whole_example_sets(tmp_path):
    from src.model_comparison.cv import fold_training_files
    from src.number_game.ablation.eliminate import ABLATION, example_set_clusters
    from src.ablation.eliminate import Args, _selection

    ng_data.main(ng_data.Args(out=tmp_path / "responses.csv"))
    args = Args(final="ng", data=tmp_path / "responses.csv", out=tmp_path / "out")
    rows, folds, paths = fold_training_files(args.data, _selection(ABLATION, args))
    units = example_set_clusters(rows)
    n_sets = len({r["number_set"] for r in rows})
    assert len(rows) == 22680 and units.max() + 1 == n_sets
    # an example set drawn by several runs is one unit, in one fold
    assert all(len(set(folds[units == u])) == 1 for u in range(n_sets))
    assert sorted(set(folds)) == [0, 1, 2, 3, 4] and np.bincount(folds).sum() == 22680
    assert all(r["number_set"] == rows[np.flatnonzero(units == units[i])[0]]["number_set"] for i, r in enumerate(rows))


# --- the rule ---------------------------------------------------------------------------

N_UNITS = 20
UNITS = np.repeat(np.arange(N_UNITS), 3)  # three responses per example set


def fake_score(contributions, unconverged=(), calls=None):
    """Each model's pointwise ELPD-CV: a base plus each mechanism's contribution per response."""
    from src.ablation.eliminate import Scored

    def score(sets):
        if calls is not None:
            calls.append([V.variant_name("ng", s) for s in sets])
        return {s: Scored(-1.0 + sum((contributions[m] for m in s), np.zeros(len(UNITS))), s not in unconverged)
                for s in sets}
    return score


RULE_START = ["last_digit", "range_low", "seed", "recency", "memory_limit"]
RULE_CONTRIBUTIONS = {
    "range_low": np.full(len(UNITS), 5.0),                  # the only range cue: never a candidate
    "last_digit": np.full(len(UNITS), 0.3),                 # small but on every set: kept
    "recency": np.repeat(np.tile([0.01, -0.01], N_UNITS // 2), 3),  # noise across sets: dropped
    "memory_limit": np.full(len(UNITS), -0.01),             # its removal is a gain: dropped first
    "seed": np.where(UNITS == 0, 0.5, 0.0),                 # 1.5 nats, all on one set: under 2 SE, dropped
}


def test_the_rule_drops_free_mechanisms_cheapest_first_and_stops_at_a_costly_one():
    from src.ablation.eliminate import eliminate

    calls = []
    record = eliminate(V.VARIANTS, "ng", RULE_START, fake_score(RULE_CONTRIBUTIONS, calls=calls), UNITS)
    decisions = [r["decision"] for r in record["rounds"]]
    assert decisions[:3] == ["drop memory_limit", "drop recency", "drop seed"]
    # The same cost on every response: the SE is 0 up to rounding, the cost-to-SE ratio huge.
    assert len(decisions) == 4 and decisions[3].startswith("stop: the cheapest removal (last_digit) costs 18.0 nats")
    # Round 1 cannot remove the last-digit cue while recency is on, nor the only range cue.
    assert {c["mechanism"] for c in record["rounds"][0]["candidates"]} == {"seed", "recency", "memory_limit"}
    assert calls[0][0] == V.variant_name("ng", RULE_START)
    seed_round = next(c for c in record["rounds"][2]["candidates"] if c["mechanism"] == "seed")
    assert seed_round["cost"] == pytest.approx(1.5) and seed_round["cost_in_se"] < 2  # clustered: one set
    assert record["lean"]["mechanisms"] == ["last_digit", "range_low"]
    assert record["lean"]["removed"] == ["seed", "recency", "memory_limit"]
    assert record["lean"]["cost_vs_start"] == pytest.approx(1.5 - 0.6)  # memory_limit cost 0.01 per response
    # intercept, slope, band_width and two saliences (one free)
    assert record["lean"]["size"] == {"population": 4, "person_vectors": 0}


def test_an_unconverged_candidate_is_never_dropped_and_an_unconverged_start_raises():
    from src.ablation.eliminate import eliminate

    no_memory = V.canonical("ng", set(RULE_START) - {"memory_limit"})
    record = eliminate(V.VARIANTS, "ng", RULE_START, fake_score(RULE_CONTRIBUTIONS, unconverged={no_memory}), UNITS)
    assert record["rounds"][0]["decision"] == "drop recency"
    assert "memory_limit" not in record["lean"]["mechanisms"]  # dropped later, from a model whose fit converged
    with pytest.raises(RuntimeError, match="did not converge"):
        eliminate(V.VARIANTS, "ng", RULE_START,
                  fake_score(RULE_CONTRIBUTIONS, unconverged={V.canonical("ng", RULE_START)}), UNITS)


def test_an_array_task_fits_its_share_of_the_rounds_tasks(tmp_path, monkeypatch):
    from src.number_game.ablation import eliminate as ng

    (tmp_path / "next.json").write_text(json.dumps(dict(status="need", tasks=[{}] * 7, seeds=[1], attempt=1)))
    done = []
    monkeypatch.setattr(ng.E, "worker", lambda ablation, args: done.append(args.task))
    args = ng.Args(final="ng", data=RUN1, out=tmp_path, mode="worker", tasks_per_job=5)
    ng.worker(ng.replace(args, task=0))
    ng.worker(ng.replace(args, task=1))
    assert done == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="has no tasks"):
        ng.worker(ng.replace(args, task=2))


# --- plan and workers, end to end ---------------------------------------------------------


@pytest.mark.slow
def test_plan_and_workers_run_the_elimination_to_a_lean_model(tmp_path):
    from dataclasses import replace

    from src.number_game.ablation.eliminate import Args, plan, worker

    data = tmp_path / "data" / "responses.csv"
    data.parent.mkdir()
    lines = RUN1.read_text().splitlines()
    data.write_text("\n".join([lines[0]] + lines[1::4]) + "\n")
    # Small and well identified at these short settings (the two range cues alone are not:
    # their salience split and the band width trade off on a quarter of run 1's data).
    start = ["parity", "range_high", "memory_limit"]
    args = Args(final="ng", data=data, out=tmp_path / "ng", start=start, folds=2,
                draws=300, tune=300, chains=2, target_accept=0.9, method="nuts", seeds=[1], time_limit_hours=0.5,
                tasks_per_job=3)
    for _ in range(4):
        doc = plan(args)
        if doc["status"] == "done":
            break
        assert doc["attempt"] == 1 and {t["fold"] for t in doc["tasks"]} == {0, 1}
        for i in range(-(-len(doc["tasks"]) // args.tasks_per_job)):
            worker(replace(args, mode="worker", task=i))
    assert doc["status"] == "done"
    record = json.loads((args.out / "elimination.json").read_text())
    assert record["rounds"][0]["model"] == V.variant_name("ng", start)
    assert len(record["rounds"][0]["candidates"]) == 2
    assert all(np.isfinite(c["elpd_cv"]) for r in record["rounds"] for c in r["candidates"])
    lean = record["lean"]["model"]
    assert (args.out / "lean" / f"{lean}.py").exists() and (args.out / "summary.md").exists()
    fold = json.loads((args.out / "cv" / record["rounds"][0]["model"] / "fold0.json").read_text())
    assert fold["converged"] and [a["seed"] for a in fold["attempts"]] == [1]
    with pytest.raises(ValueError, match="another configuration"):
        plan(replace(args, draws=301))


def test_eliminate_imports_no_arviz():
    """A fit child re-imports the script it was started from (``python -m ...eliminate``)
    and refuses to fit when arviz is already loaded (tests/test_sr_ablation.py)."""
    import subprocess
    import sys

    check = "import sys, src.number_game.ablation.eliminate; print('arviz' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", check], cwd=here(), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
