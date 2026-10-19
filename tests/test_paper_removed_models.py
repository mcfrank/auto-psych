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


def _entry(outcome, detail, context):
    return {"file": None, "outcome": outcome, "detail": detail, "context": context}


def test_only_models_pruned_at_the_end_of_the_final_experiment_keep_their_recorded_elpd(tmp_path):
    final = _history(tmp_path, [{"elpd_loo": {"late": -10.0, "capped": -11.0, "early": -12.0}},
                                {"elpd_loo": {"late": -10.0}}])
    removed = {
        "late": _entry("pruned", "4 nats behind w (3× dse)", "experiment3 end of experiment"),
        "capped": _entry("pruned", f"9 nats behind w; {rm.RETIRED_MARK} at 8",
                         "experiment3 end of experiment"),
        "early": _entry("pruned", "4 nats behind w (3× dse)", "experiment2 end of experiment"),
        "broken": _entry("dropped", "MCMC fit failed", "experiment3 round 2"),
    }
    assert rm.recorded_scores(final, removed) == {"late": -10.0}


def test_a_pruned_model_scored_differently_within_the_final_experiment_raises(tmp_path):
    final = _history(tmp_path, [{"elpd_loo": {"late": -10.0}}, {"elpd_loo": {"late": -10.5}}])
    removed = {"late": _entry("pruned", "4 nats behind w", "experiment3 end of experiment")}
    with pytest.raises(RuntimeError, match="2 distinct ELPDs"):
        rm.recorded_scores(final, removed)
