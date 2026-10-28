"""Ablating the number-game final: the switchable rewrite, its variants, the data, the elimination."""

import json

import numpy as np
import pytest
from pyprojroot import here

from src.number_game.ablation import variants as V
from src.number_game.ablation.data import RESPONSES

FINAL_DIR = here() / "data/results/number_game_dataset-oct26/collected/run1/number_game/experiment3/cognitive_models"
MECHANISMS = V.spec("ng").MECHANISMS


def bound(model):
    import pymc as pm

    from src.models.data_binding import extract_observed

    with model:
        pm.set_data(extract_observed(RESPONSES, model))
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


def at_values(model, point, values):
    """``point`` with the named free variables at these (constrained) values."""
    out = dict(point)
    for name, value in values.items():
        rv = model[name]
        transform = model.rvs_to_transforms.get(rv)
        with np.errstate(divide="ignore"):
            v = value if transform is None else transform.forward(np.float64(value), *rv.owner.inputs).eval()
        out[model.rvs_to_values[rv].name] = np.full_like(point[model.rvs_to_values[rv].name], v)
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
    assert V.invalid_reason("ng", off) is None and V.invalid_reason("ng", off | {mechanism}) is None
    with_m = variant_model(off | {mechanism}, tmp_path)
    without_m = variant_model(off, tmp_path)
    kept = {v.name for v in without_m.value_vars}
    assert kept <= {v.name for v in with_m.value_vars}  # removing a mechanism removes parameters or structure only
    point = jittered(with_m, 0)
    at = at_values(with_m, point, V.spec("ng").NEUTRAL[mechanism])
    assert np.allclose(p_yes(with_m, at), p_yes(without_m, {k: at[k] for k in kept}), rtol=0, atol=1e-10)
    # ... and away from it, the mechanism moves some prediction.
    assert np.abs(p_yes(with_m, point) - p_yes(without_m, {k: point[k] for k in kept})).max() > 1e-4


def test_a_modifier_goes_before_the_term_it_modifies_and_one_term_stays():
    first = V.removals("ng", MECHANISMS)
    assert {"parity_preempt", "magnitude_preempt", "count_step_cost", "intercept", "even_rule"} <= set(first)
    assert not {"parity", "decade_band", "counting"} & set(first)
    assert "decade_band" in V.removals("ng", set(MECHANISMS) - {"magnitude_preempt"})
    # Parity credit acts only through the similarity and the mutual standard.
    assert V.invalid_reason("ng", ["parity", "intercept"]) == "parity needs one of ['similarity', 'mutual_standard']"
    assert V.removals("ng", ["similarity", "parity"]) == ["parity"]
    # Some term stays: an empty model has nothing to fit.
    assert V.removals("ng", ["even_rule"]) == []
    assert V.invalid_reason("ng", []).startswith("one of ['intercept', 'similarity'")
    with pytest.raises(ValueError, match="unknown mechanisms"):
        V.canonical("ng", ["intercept", "telepathy"])


def test_variant_files_are_self_contained_and_named_by_what_was_removed(tmp_path):
    name, path = V.write_variant("ng", set(MECHANISMS) - {"even_rule", "parity_preempt"}, tmp_path)
    assert name == "ng_minus_parpre_even"
    assert "def build(on):" in path.read_text() and path.read_text().endswith("model = build(MECHANISMS_ON)\n")
    assert "# --- this variant (src.number_game.ablation.variants) ---" in path.read_text()
    with pytest.raises(ValueError, match="is not a model"):
        V.write_variant("ng", set(MECHANISMS) - {"similarity", "mutual_standard"}, tmp_path)


def test_the_full_variant_has_the_finals_parameter_count():
    assert V.size("ng", MECHANISMS) == {"population": 14, "person_vectors": 0}


def test_the_full_and_the_smallest_variant_meet_the_data_contract(tmp_path):
    from src.models.model_contract import model_contract_violation

    smallest = set(MECHANISMS)
    while removable := V.removals("ng", smallest):
        smallest -= {removable[0]}
    assert len(smallest) == 1
    for on in (MECHANISMS, smallest):
        name, _ = V.write_variant("ng", on, tmp_path)
        assert model_contract_violation(name, tmp_path, RESPONSES) is None, name


# --- the data and the CV units -----------------------------------------------------------


def test_the_data_are_the_runs_own_responses(tmp_path):
    from src.number_game.ablation import data

    lines = RESPONSES.read_text().splitlines()
    assert lines[0] == "number_set,target,trial_index,rating" and len(lines) - 1 == 7560
    data.main(data.Args(out=tmp_path / "responses.csv"))
    data.main(data.Args(out=tmp_path / "responses.csv"))  # up to date: kept
    (tmp_path / "responses.csv").write_text(RESPONSES.read_text() + '"2,4",6,0,1\n')
    with pytest.raises(ValueError, match="is not"):
        data.main(data.Args(out=tmp_path / "responses.csv"))


def test_folds_and_clusters_are_whole_example_sets(tmp_path):
    import shutil

    from src.model_comparison.cv import fold_training_files
    from src.number_game.ablation.eliminate import ABLATION, example_set_clusters
    from src.ablation.eliminate import Args, _selection

    shutil.copy(RESPONSES, tmp_path / "responses.csv")
    args = Args(final="ng", data=tmp_path / "responses.csv", out=tmp_path / "out")
    rows, folds, paths = fold_training_files(args.data, _selection(ABLATION, args))
    units = example_set_clusters(rows)
    assert units.max() + 1 == len({r["number_set"] for r in rows}) == 150
    assert all(len(set(folds[units == u])) == 1 for u in range(150))  # an example set is in one fold
    assert list(np.bincount(folds)) == [1512] * 5
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


RULE_START = ["intercept", "counting", "count_step_cost", "even_rule", "size_principle"]
RULE_CONTRIBUTIONS = {
    "intercept": np.full(len(UNITS), 5.0),                  # large: kept
    "counting": np.full(len(UNITS), 0.3),                   # small but on every set: kept
    "count_step_cost": np.repeat(np.tile([0.01, -0.01], N_UNITS // 2), 3),  # noise across sets: dropped
    "even_rule": np.full(len(UNITS), -0.01),                # its removal is a gain: dropped first
    "size_principle": np.where(UNITS == 0, 0.5, 0.0),       # 1.5 nats, all on one set: under 2 SE, dropped
}


def test_the_rule_drops_free_mechanisms_cheapest_first_and_stops_at_a_costly_one():
    from src.ablation.eliminate import eliminate

    calls = []
    record = eliminate(V.VARIANTS, "ng", RULE_START, fake_score(RULE_CONTRIBUTIONS, calls=calls), UNITS)
    decisions = [r["decision"] for r in record["rounds"]]
    assert decisions[:3] == ["drop even_rule", "drop count_step_cost", "drop size_principle"]
    # The same cost on every response: the SE is 0 up to rounding, the cost-to-SE ratio huge.
    assert len(decisions) == 4 and decisions[3].startswith("stop: the cheapest removal (counting) costs 18.0 nats")
    # Round 1 cannot remove counting while its step cost is on.
    assert {c["mechanism"] for c in record["rounds"][0]["candidates"]} == \
        {"intercept", "count_step_cost", "even_rule", "size_principle"}
    assert calls[0][0] == "ng_minus_sim_par_parpre_std_cf_round_band_magpre"
    size_round = next(c for c in record["rounds"][2]["candidates"] if c["mechanism"] == "size_principle")
    assert size_round["cost"] == pytest.approx(1.5) and size_round["cost_in_se"] < 2  # clustered: one set
    assert record["lean"]["mechanisms"] == ["intercept", "counting"]
    assert record["lean"]["removed"] == ["count_step_cost", "size_principle", "even_rule"]
    assert record["lean"]["cost_vs_start"] == pytest.approx(1.5 - 0.6)  # even_rule cost 0.01 per response
    assert record["lean"]["size"] == {"population": 2, "person_vectors": 0}


def test_an_unconverged_candidate_is_never_dropped_and_an_unconverged_start_raises():
    from src.ablation.eliminate import eliminate

    no_even = V.canonical("ng", set(RULE_START) - {"even_rule"})
    record = eliminate(V.VARIANTS, "ng", RULE_START, fake_score(RULE_CONTRIBUTIONS, unconverged={no_even}), UNITS)
    assert record["rounds"][0]["decision"] == "drop count_step_cost"
    assert "even_rule" not in record["lean"]["mechanisms"]  # dropped later, from a model whose fit converged
    with pytest.raises(RuntimeError, match="did not converge"):
        eliminate(V.VARIANTS, "ng", RULE_START,
                  fake_score(RULE_CONTRIBUTIONS, unconverged={V.canonical("ng", RULE_START)}), UNITS)


def test_an_array_task_fits_its_share_of_the_rounds_tasks(tmp_path, monkeypatch):
    from src.number_game.ablation import eliminate as ng

    (tmp_path / "next.json").write_text(json.dumps(dict(status="need", tasks=[{}] * 7, seeds=[1], attempt=1)))
    done = []
    monkeypatch.setattr(ng.E, "worker", lambda ablation, args: done.append(args.task))
    args = ng.Args(final="ng", data=RESPONSES, out=tmp_path, mode="worker", tasks_per_job=5)
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
    lines = RESPONSES.read_text().splitlines()
    data.write_text("\n".join([lines[0]] + lines[1::4]) + "\n")
    args = Args(final="ng", data=data, out=tmp_path / "ng", start=["intercept", "even_rule"], folds=2,
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
    assert record["rounds"][0]["model"] == "ng_minus_sim_par_parpre_std_cf_round_band_magpre_count_step_size"
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
