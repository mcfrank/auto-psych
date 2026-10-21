"""Ablating the sr-oct26 finals: the switchable rewrites, the variants, the elimination rule."""

import json

import numpy as np
import pytest
from pyprojroot import here

from src.subjective_randomness.ablation import variants as V

COLLECTED = here() / "data/results/human_experiment_sr-oct26/collected"
FINAL_DIRS = {f"run{r}": COLLECTED / f"run{r}/subjective_randomness/experiment3/model_loop" for r in (1, 2, 3)}
CASES = [(f, m) for f in V.FINALS for m in V.spec(f).MECHANISMS]


@pytest.fixture(scope="module")
def responses(tmp_path_factory):
    """A slice of every run's final data: every stimulus pair of each run, a few people."""
    import csv

    path = tmp_path_factory.mktemp("sr_ablation") / "responses.csv"
    rows = []
    for d in FINAL_DIRS.values():
        with (d / "responses.csv").open(newline="") as f:
            rows += [r for r in csv.DictReader(f) if int(r["participant_id"]) < 6]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def bound(model, responses):
    import pymc as pm

    from src.models.data_binding import extract_observed

    with model:
        pm.set_data(extract_observed(responses, model))
    return model


def variant_model(final, on, out_dir, responses):
    from src.models.model_loading import load_pymc_model

    name, _ = V.write_variant(final, on, out_dir)
    return bound(load_pymc_model(name, out_dir), responses)


def jittered(model, seed):
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, 0.5, size=np.shape(v)) for k, v in model.initial_point().items()}


def p_left(model, point):
    # p_left is a function of the random variables: on their values, not on fresh draws.
    out = model.replace_rvs_by_values([model["p_left"]])[0]
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


def without(final, m):
    """The largest valid set without ``m`` (every mechanism that needs it removed too)."""
    requires = V.spec(final).REQUIRES
    on = set(V.spec(final).MECHANISMS) - {m}
    while unmet := {x for x in on if x in requires and not on & set(requires[x])}:
        on -= unmet
    return on


@pytest.mark.parametrize("final", list(V.FINALS))
def test_with_every_mechanism_on_the_variant_is_the_final(final, responses, tmp_path):
    from src.models.model_loading import load_pymc_model

    s = V.spec(final)
    original = bound(load_pymc_model(s.FINAL, FINAL_DIRS[final] / "models"), responses)
    mine = variant_model(final, s.MECHANISMS, tmp_path, responses)
    assert V.variant_name(final, s.MECHANISMS) == f"{V.PREFIX[final]}_full"
    assert {(v.name, tuple(original.initial_point()[v.name].shape)) for v in original.value_vars} == \
        {(v.name, tuple(mine.initial_point()[v.name].shape)) for v in mine.value_vars}
    logp_o, logp_m = original.compile_logp(), mine.compile_logp()
    for seed in range(4):
        point = jittered(original, seed)
        assert logp_m(point) == pytest.approx(logp_o(point), rel=1e-10, abs=1e-8)
        assert np.allclose(p_left(mine, point), p_left(original, point), rtol=0, atol=1e-12)


@pytest.mark.parametrize("final,mechanism", CASES)
def test_each_mechanism_at_its_neutral_value_is_the_model_without_it(final, mechanism, responses, tmp_path):
    off = without(final, mechanism)
    assert V.invalid_reason(final, off) is None and V.invalid_reason(final, off | {mechanism}) is None
    with_m = variant_model(final, off | {mechanism}, tmp_path, responses)
    without_m = variant_model(final, off, tmp_path, responses)
    kept = {v.name for v in without_m.value_vars}
    assert kept <= {v.name for v in with_m.value_vars}  # removing a mechanism removes parameters or structure only
    point = jittered(with_m, 0)
    neutral = V.spec(final).NEUTRAL[mechanism]
    if neutral is not None:
        at = at_values(with_m, point, neutral)
        assert np.allclose(p_left(with_m, at), p_left(without_m, {k: at[k] for k in kept}), rtol=0, atol=1e-10)
    # ... and away from it, the mechanism moves some prediction.
    assert np.abs(p_left(with_m, point) - p_left(without_m, {k: point[k] for k in kept})).max() > 1e-4


def test_person_level_variation_goes_before_the_parameter_it_varies():
    full = V.spec("run1").MECHANISMS
    first = V.removals("run1", full)
    assert {"imbalance_personal", "lapse_personal", "gain_personal", "final_run"} <= set(first)
    assert not {"imbalance", "longest_run", "lapse"} & set(first)
    assert "imbalance" in V.removals("run1", set(full) - {"imbalance_personal"})
    # The pattern gain needs some structural cue to scale.
    structure = set(V.spec("run1").STRUCTURE)
    assert V.invalid_reason("run1", set(full) - structure) == f"gain_personal needs one of {list(V.spec('run1').STRUCTURE)}"
    assert "antisymmetry" not in V.removals("run1", (set(full) - structure) | {"antisymmetry"})
    # Run 3 keeps a regular generator.
    assert V.removals("run3", ["markov", "side_bias"]) == ["side_bias"]
    assert V.invalid_reason("run3", ["side_bias"]) == "one of ['markov', 'trick_coin', 'motif'] must stay"
    with pytest.raises(ValueError, match="unknown mechanisms"):
        V.canonical("run2", ["switch", "telepathy"])


def test_variant_files_are_self_contained_and_named_by_what_was_removed(tmp_path):
    full = set(V.spec("run1").MECHANISMS)
    name, path = V.write_variant("run1", full - {"variety", "palindrome"}, tmp_path)
    assert name == "r1_minus_pal_var"
    assert "def build(on):" in path.read_text() and path.read_text().endswith("model = build(MECHANISMS_ON)\n")
    assert V.write_variant("run1", full - {"palindrome", "variety"}, tmp_path) == (name, path)
    path.write_text(path.read_text() + "# edited\n")
    with pytest.raises(ValueError, match="another version"):
        V.write_variant("run1", full - {"variety", "palindrome"}, tmp_path)
    with pytest.raises(ValueError, match="is not a model"):
        V.write_variant("run1", full - {"imbalance"}, tmp_path)


@pytest.mark.parametrize("final,counts", [
    # Table 1 of the paper (k + n p): 17 + 6p, 15 + 6p, 14 + 4p.
    ("run1", {"population": 17, "person_vectors": 6}),
    ("run2", {"population": 15, "person_vectors": 6}),
    ("run3", {"population": 14, "person_vectors": 4}),
])
def test_the_full_variants_have_the_finals_parameter_counts(final, counts):
    assert V.size(final, V.spec(final).MECHANISMS) == counts


@pytest.mark.parametrize("final", list(V.FINALS))
def test_the_full_and_the_smallest_variant_meet_the_data_contract(final, responses, tmp_path):
    from src.models.model_contract import model_contract_violation

    smallest = set(V.spec(final).MECHANISMS)
    while removable := V.removals(final, smallest):
        smallest -= {removable[0]}
    for on in (V.spec(final).MECHANISMS, smallest):
        name, _ = V.write_variant(final, on, tmp_path)
        assert model_contract_violation(name, tmp_path, responses) is None, name


# --- the pooled data -------------------------------------------------------------------


def test_the_pooled_data_are_the_papers_pooled_fits_data(tmp_path):
    from scripts.paper.sr_figure_data import pool_responses
    from src.subjective_randomness.ablation import pool

    theirs = tmp_path / "theirs.csv"
    pool_responses({r: pool.run_responses(r) for r in pool.RUNS}, theirs)
    text = pool.pooled_text()
    assert text == theirs.read_text()
    lines = text.splitlines()[1:]
    assert len(lines) == 23104 and len({line.split(",")[2] for line in lines}) == 359
    pool.main(pool.Args(out=tmp_path / "pooled.csv"))
    pool.main(pool.Args(out=tmp_path / "pooled.csv"))  # up to date: kept
    (tmp_path / "pooled.csv").write_text(text + "HT,TH,0,1,1\n")
    with pytest.raises(ValueError, match="is not the pooled data"):
        pool.main(pool.Args(out=tmp_path / "pooled.csv"))


# --- the rule ---------------------------------------------------------------------------

N_UNITS = 20
UNITS = np.repeat(np.arange(N_UNITS), 2)  # two trials per stimulus pair


def fake_score(final, contributions, unconverged=(), calls=None):
    """Each model's pointwise ELPD-CV: a base plus each mechanism's contribution per trial."""
    from src.subjective_randomness.ablation.eliminate import Scored

    def score(sets):
        if calls is not None:
            calls.append([V.variant_name(final, s) for s in sets])
        return {s: Scored(-1.0 + sum((contributions[m] for m in s), np.zeros(len(UNITS))), s not in unconverged)
                for s in sets}
    return score


RULE_START = ["switch", "switch_ideal_personal", "variety", "final_run"]
RULE_CONTRIBUTIONS = {
    "switch": np.full(len(UNITS), 5.0),                    # large: kept
    "switch_ideal_personal": np.tile([0.01, -0.01], N_UNITS),  # nothing on any pair: dropped
    "variety": np.full(len(UNITS), -0.01),                 # its removal is a gain: dropped first
    "final_run": np.full(len(UNITS), 0.5),                 # small but on every pair: kept
}


def test_the_rule_drops_free_mechanisms_cheapest_first_and_stops_at_a_costly_one():
    from src.subjective_randomness.ablation.eliminate import eliminate

    calls = []
    record = eliminate("run2", RULE_START, fake_score("run2", RULE_CONTRIBUTIONS, calls=calls), UNITS)
    assert [r["decision"] for r in record["rounds"]] == [
        "drop variety", "drop switch_ideal_personal",
        "stop: the cheapest removal (final_run) costs 20.0 nats, inf SE >= 2.0"]
    # Round 1 cannot remove the switch term while its personal ideal is on.
    assert {c["mechanism"] for c in record["rounds"][0]["candidates"]} == {"switch_ideal_personal", "variety", "final_run"}
    assert calls[0][0] == "r2_minus_idealP_betaP_len_swweightP_rule_patP_open_sideP"
    assert record["lean"]["mechanisms"] == ["switch", "final_run"]
    assert record["lean"]["removed"] == ["switch_ideal_personal", "variety"]
    assert record["lean"]["cost_vs_start"] == pytest.approx(-0.4)  # variety cost 0.01 per trial: removing it gained
    assert record["lean"]["size"] == {"population": 5, "person_vectors": 0}  # the core 2, switch 2, final_run 1


def test_an_unconverged_candidate_is_never_dropped_and_an_unconverged_start_raises():
    from src.subjective_randomness.ablation.eliminate import eliminate

    no_variety = V.canonical("run2", set(RULE_START) - {"variety"})
    record = eliminate("run2", RULE_START, fake_score("run2", RULE_CONTRIBUTIONS, unconverged={no_variety}), UNITS)
    assert record["rounds"][0]["decision"] == "drop switch_ideal_personal"
    assert "variety" not in record["lean"]["mechanisms"]  # dropped later, from a model whose fit converged
    with pytest.raises(RuntimeError, match="did not converge"):
        eliminate("run2", RULE_START, fake_score("run2", RULE_CONTRIBUTIONS, unconverged={V.canonical("run2", RULE_START)}),
                  UNITS)


def test_a_fold_is_needed_until_it_converges_or_every_seed_was_tried():
    from src.subjective_randomness.ablation.eliminate import _fold_needed

    assert _fold_needed(None, [1])
    unconverged = dict(converged=False, attempts=[dict(seed=1)])
    assert not _fold_needed(unconverged, [1])
    assert _fold_needed(unconverged, [1, 2])  # seeds added: tried again at 2
    assert not _fold_needed(dict(converged=True, attempts=[dict(seed=1)]), [1, 2])


# --- plan and workers, end to end ---------------------------------------------------------


@pytest.mark.slow
def test_plan_and_workers_run_the_elimination_to_a_lean_model(tmp_path):
    from dataclasses import replace

    from src.subjective_randomness.ablation import pool
    from src.subjective_randomness.ablation.eliminate import Args, plan, worker

    lines = pool.pooled_text().splitlines()
    data = tmp_path / "data" / "responses.csv"
    data.parent.mkdir()
    data.write_text("\n".join([lines[0]] + [x for x in lines[1:] if int(x.split(",")[2]) % 60 == 0]) + "\n")
    args = Args(final="run2", data=data, out=tmp_path / "run2", start=["final_run", "opening_run"], folds=2,
                draws=300, tune=300, chains=2, target_accept=0.9, method="nuts", seeds=[1], time_limit_hours=0.5)
    for _ in range(4):
        doc = plan(args)
        if doc["status"] == "done":
            break
        assert doc["attempt"] == 1 and {t["fold"] for t in doc["tasks"]} == {0, 1}
        for i in range(len(doc["tasks"])):
            worker(replace(args, mode="worker", task=i))
    assert doc["status"] == "done"
    record = json.loads((args.out / "elimination.json").read_text())
    assert record["rounds"][0]["model"] == "r2_minus_idealP_betaP_len_sw_swidealP_swweightP_var_rule_patP_sideP"
    assert len(record["rounds"][0]["candidates"]) == 2
    assert all(np.isfinite(c["elpd_cv"]) for r in record["rounds"] for c in r["candidates"])
    lean = record["lean"]["model"]
    assert (args.out / "lean" / f"{lean}.py").exists() and (args.out / "summary.md").exists()
    fold = json.loads((args.out / "cv" / record["rounds"][0]["model"] / "fold0.json").read_text())
    assert fold["converged"] and [a["seed"] for a in fold["attempts"]] == [1]
    with pytest.raises(ValueError, match="another configuration"):
        plan(replace(args, draws=301))
