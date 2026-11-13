"""``scripts/paper/removed_models.py``: the removed models of a run scored in
shards, one record per model, and collected for the figure data."""

from __future__ import annotations

import pytest

from scripts.paper import removed_models as rm
from src.models import pymc_inference


def test_the_shards_partition_the_items():
    items = list(range(10))
    shards = [rm.shard_of(items, k, 3) for k in range(3)]
    assert sorted(sum(shards, [])) == items
    assert shards[0] == [0, 3, 6, 9]
    with pytest.raises(ValueError, match="not in 0..2"):
        rm.shard_of(items, 3, 3)


def _files(tmp_path, names):
    files = {}
    for name in names:
        files[name] = tmp_path / "src" / f"{name}.py"
        files[name].parent.mkdir(exist_ok=True)
        files[name].write_text(f"# {name}\n")
    return files


def _inputs(name):
    return {"model": name, "settings": 1}


def test_a_shard_records_each_model_and_a_failed_fit_as_unscored(tmp_path, monkeypatch):
    files = _files(tmp_path, ["a", "b", "c"])
    fitted_on = []

    def fake_fit(names, models_dir, path, *, cache_dir, **kwargs):
        fitted_on.append((path.name, list(names)))
        return {"b": "RuntimeError: boom"} if path.name == "fold0.csv" else {}

    monkeypatch.setattr(pymc_inference, "fit_models_to_cache", fake_fit)
    records = tmp_path / "records"
    kwargs = dict(models_dir=tmp_path / "models", records_dir=records, inputs=_inputs,
                  training_files=[("all data", tmp_path / "all.csv"),
                                  ("CV fold 0", tmp_path / "fold0.csv"),
                                  ("CV fold 1", tmp_path / "fold1.csv")],
                  score=lambda n: {"elpd": -float(ord(n))},
                  cache_dir=tmp_path / "cache", fit_kwargs={})

    rm.score_shard(files, **kwargs)

    # A model that failed on one file is not fitted on the next.
    assert fitted_on == [("all.csv", ["a", "b", "c"]), ("fold0.csv", ["a", "b", "c"]),
                         ("fold1.csv", ["a", "c"])]
    collected = rm.collect(["a", "b", "c"], records_dir=records, inputs=_inputs)
    assert collected["elpd"] == {"a": -97.0, "c": -99.0}
    assert collected["unscored"] == {"b": "fit failed on CV fold 0 (RuntimeError: boom)"}

    # A rerun finds every record current and fits nothing.
    fitted_on.clear()
    rm.score_shard(files, **kwargs)
    assert fitted_on == []


def test_collecting_raises_naming_a_model_without_a_current_record(tmp_path, monkeypatch):
    files = _files(tmp_path, ["a"])
    monkeypatch.setattr(pymc_inference, "fit_models_to_cache", lambda *a, **k: {})
    rm.score_shard(files, models_dir=tmp_path / "models", records_dir=tmp_path / "records",
                   inputs=_inputs, training_files=[("all data", tmp_path / "all.csv")],
                   score=lambda n: {"elpd": 1.0}, cache_dir=tmp_path / "cache", fit_kwargs={})

    with pytest.raises(RuntimeError, match=r"1 of 2 removed models .*\['z'\]"):
        rm.collect(["a", "z"], records_dir=tmp_path / "records", inputs=_inputs)
    # A record made from other inputs is not current.
    with pytest.raises(RuntimeError, match=r"\['a'\]"):
        rm.collect(["a"], records_dir=tmp_path / "records",
                   inputs=lambda n: {"model": n, "settings": 2})


def _history(tmp_path, steps):
    final = tmp_path / "experiment3"
    (final / "model_loop").mkdir(parents=True)
    (final / "model_loop" / "history.json").write_text(__import__("json").dumps(steps))
    return final


def _entry(name, outcome, detail, context):
    return {"name": name, "file": None, "outcome": outcome, "detail": detail, "context": context}


def test_only_models_pruned_at_the_end_of_the_final_experiment_keep_their_recorded_elpd(tmp_path):
    final = _history(tmp_path, [{"elpd_loo": {"late": -10.0, "capped": -11.0, "early": -12.0}},
                                {"elpd_loo": {"late": -10.0}}])
    removed = {
        "late": _entry("late", "pruned", "4 nats behind w (3× dse)", "experiment3 end of experiment"),
        "capped": _entry("capped", "pruned", f"9 nats behind w; {rm.RETIRED_MARK} at 8",
                         "experiment3 end of experiment"),
        "early": _entry("early", "pruned", "4 nats behind w (3× dse)", "experiment2 end of experiment"),
        "broken": _entry("broken", "dropped", "MCMC fit failed", "experiment3 round 2"),
    }
    assert rm.recorded_scores(final, removed) == {"late": -10.0}


def test_a_pruned_model_scored_differently_within_the_final_experiment_raises(tmp_path):
    final = _history(tmp_path, [{"elpd_loo": {"late": -10.0}}, {"elpd_loo": {"late": -10.5}}])
    removed = {"late": _entry("late", "pruned", "4 nats behind w", "experiment3 end of experiment")}
    with pytest.raises(RuntimeError, match="2 distinct ELPDs"):
        rm.recorded_scores(final, removed)


def _ledger_run(tmp_path, events, files):
    """A run tree: the final experiment's ledger (``events``: name, outcome,
    context) and each model file at ``experiment<N>/model_loop/models[/pruned]``."""
    import json

    run = tmp_path / "run"
    final = run / "experiment3"
    (final / "model_loop").mkdir(parents=True)
    (final / "model_loop" / "attempted_hypotheses.jsonl").write_text("".join(
        json.dumps({"name": n, "outcome": o, "detail": "d", "hypothesis": "h", "context": c}) + "\n"
        for n, o, c in events))
    for rel, text in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(text)
    return run, final


def test_a_name_admitted_twice_with_different_code_is_two_removed_models(tmp_path):
    """Number-game run 3 (nov26) admitted two different models under the
    placeholder name iter2_candidate1, in experiments 1 and 3: each is its own
    removed model, keyed by its experiment, with its own file and outcome."""
    run, final = _ledger_run(tmp_path, [
        ("w", "admitted", "experiment1 round 0 candidate 0"),
        ("x", "admitted", "experiment1 round 2 candidate 1"),
        ("x", "pruned", "experiment1 end of experiment"),
        ("y", "admitted", "experiment2 round 0 candidate 0"),
        ("y", "pruned", "experiment2 end of experiment"),
        ("x", "admitted", "experiment3 round 2 candidate 1"),
        ("x", "pruned", "experiment3 end of experiment"),
    ], {
        "experiment3/model_loop/models/w.py": "w",
        "experiment1/model_loop/models/pruned/x.py": "x, first",
        "experiment3/model_loop/models/pruned/x.py": "x, second",
        "experiment2/model_loop/models/pruned/y.py": "y",
    })

    removed = rm.removed_models(run, final, live=["w"], seeds=[])

    assert list(removed) == ["x_experiment1", "y", "x_experiment3"]
    assert removed["x_experiment1"]["file"] == run / "experiment1/model_loop/models/pruned/x.py"
    assert removed["x_experiment3"]["file"] == run / "experiment3/model_loop/models/pruned/x.py"
    assert removed["x_experiment1"]["context"] == "experiment1 end of experiment"
    assert removed["x_experiment3"]["context"] == "experiment3 end of experiment"
    assert {k: v["name"] for k, v in removed.items()} == {
        "x_experiment1": "x", "y": "y", "x_experiment3": "x"}
    assert rm.admission_order(final) == {"w": 0, "x_experiment1": 1, "y": 2, "x_experiment3": 3}
    # Only the experiment-3 copy was pruned at the end of the final experiment,
    # where history.json records its ELPD under the ledger's name.
    (final / "model_loop" / "history.json").write_text('[{"elpd_loo": {"x": -7.0}}]')
    assert rm.recorded_scores(final, removed) == {"x_experiment3": -7.0}


def test_a_name_admitted_twice_with_the_same_code_stays_one_model(tmp_path):
    run, final = _ledger_run(tmp_path, [
        ("x", "admitted", "experiment1 round 2 candidate 1"),
        ("x", "pruned", "experiment1 end of experiment"),
        ("x", "admitted", "experiment2 round 0 candidate 0"),
        ("x", "pruned", "experiment2 end of experiment"),
    ], {
        "experiment1/model_loop/models/pruned/x.py": "x",
        "experiment2/model_loop/models/pruned/x.py": "x",
    })

    removed = rm.removed_models(run, final, live=[], seeds=[])

    assert list(removed) == ["x"]
    assert removed["x"]["context"] == "experiment2 end of experiment"
    assert rm.admission_order(final) == {"x": 0}
