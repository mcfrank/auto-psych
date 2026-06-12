"""CLI-parsing tests for the subjective-randomness recovery scripts.

These pin the argument layer of the script entry points (which live under
`scripts/`, not as importable packages) so the optional `--tidy-csv` output
flag cannot silently regress. The recovery business logic is tested elsewhere.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "subjective_randomness"

# Canned runner outputs so the --tidy-csv branch can be exercised end-to-end
# without running any optimizer/MCMC (the expensive runner is monkeypatched).
PARAM_REPORT = {
    "model": "demo",
    "n_repeats": 2,
    "n_stimuli": 4,
    "true_params": {"beta": 4.0},
    "runs": [
        {"repeat": 0, "posterior": {"beta": {"mean": 4.2, "q025": 3.2, "q975": 5.2}}},
        {"repeat": 1, "posterior": {"beta": {"mean": 3.8, "q025": 2.8, "q975": 4.8}}},
    ],
}
CONFUSION_RESULT = {
    "seed_models": ["A", "B"],
    "generator": "pymc",
    "n_participants": 5,
    "n_stimuli": 4,
    "fit_kwargs": {},
    "generating": [
        {
            "generating_model": "A",
            "params": {},
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.8, "B": 0.2},
            "elpd_loo": {"A": -1.0, "B": -2.0},
        }
    ],
}
PARAM_TIDY_COLUMNS = {"model", "parameter", "repeat", "true_value", "estimate", "error"}
CONFUSION_TIDY_COLUMNS = {
    "generating_model",
    "recovered_model",
    "posterior",
    "elpd_loo",
    "is_true_model",
    "is_best_model",
}


def _load_script(name: str):
    path = SCRIPTS / name
    mod_name = f"_sr_script_{name[:-3]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass Args can resolve its own module
    # (dataclasses looks the class's module up in sys.modules).
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pymc_recover_cli_tidy_csv_defaults_none_and_parses_path():
    args_cls = _load_script("pymc_recover.py").Args

    default = tyro.cli(args_cls, args=["--config", "c.yaml", "--out", "r.json"])
    assert default.tidy_csv is None

    with_tidy = tyro.cli(
        args_cls,
        args=["--config", "c.yaml", "--out", "r.json", "--tidy-csv", "t.csv"],
    )
    assert with_tidy.tidy_csv == Path("t.csv")


def test_model_recovery_cli_defaults_and_overrides():
    args_cls = _load_script("model_recovery.py").Args

    default = tyro.cli(args_cls, args=["--config", "c.yaml", "--out", "r.json"])
    assert default.config == Path("c.yaml")
    assert default.out == Path("r.json")
    assert default.tidy_csv is None
    assert default.results_root is None
    assert default.n_participants is None
    assert default.draws is None  # falls back to the config's fit settings
    assert default.generator is None  # falls back to the config's generator

    full = tyro.cli(
        args_cls,
        args=[
            "--config", "c.yaml",
            "--out", "r.json",
            "--tidy-csv", "c.csv",
            "--n-participants", "12",
            "--draws", "300",
            "--chains", "4",
            "--generator", "model_family",
        ],
    )
    assert full.tidy_csv == Path("c.csv")
    assert full.n_participants == 12
    assert full.draws == 300
    assert full.chains == 4
    assert full.generator == "model_family"


def test_analyze_recovery_cli_defaults_and_paths():
    args_cls = _load_script("analyze_recovery.py").Args

    default = tyro.cli(args_cls, args=["--results", "r.json"])
    assert default.results == Path("r.json")
    assert default.out_csv is None
    assert default.figure is None

    full = tyro.cli(
        args_cls,
        args=["--results", "r.json", "--out-csv", "s.csv", "--figure", "f.png"],
    )
    assert full.out_csv == Path("s.csv")
    assert full.figure == Path("f.png")


def test_analyze_recovery_detect_kind_classifies_both_shapes_and_rejects_other():
    detect = _load_script("analyze_recovery.py")._detect_kind

    assert detect({"generating": [], "seed_models": []}) == "model_recovery"
    assert detect({"runs": [], "true_params": {}}) == "parameter_recovery"
    # Sampled-truth reports carry param_ranges instead of a fixed true_params.
    assert detect({"runs": [], "param_ranges": {}}) == "parameter_recovery"
    with pytest.raises(ValueError, match="Unrecognized results file"):
        detect({"something": "else"})


# ── --tidy-csv branch executes (would fail if the branch were reverted) ──


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("model: demo\n", encoding="utf-8")
    return config


def test_pymc_recover_writes_tidy_csv_when_flag_set(tmp_path, monkeypatch):
    mod = _load_script("pymc_recover.py")
    monkeypatch.setattr(mod, "run_pymc_recovery", lambda *a, **k: PARAM_REPORT)
    tidy = tmp_path / "tidy.csv"

    mod.main(
        mod.Args(config=_write_config(tmp_path), out=tmp_path / "r.json", tidy_csv=tidy)
    )

    assert tidy.exists()
    rows = list(csv.DictReader(tidy.open(encoding="utf-8")))
    assert set(rows[0]) == PARAM_TIDY_COLUMNS


def test_pymc_recover_skips_tidy_csv_when_flag_absent(tmp_path, monkeypatch):
    mod = _load_script("pymc_recover.py")
    monkeypatch.setattr(mod, "run_pymc_recovery", lambda *a, **k: PARAM_REPORT)

    mod.main(mod.Args(config=_write_config(tmp_path), out=tmp_path / "r.json"))

    assert list(tmp_path.glob("*.csv")) == []  # no tidy CSV produced


def test_pymc_recover_cli_honors_config_mcmc_block(tmp_path, monkeypatch):
    # The README promises sampler settings come from the config's `mcmc` block,
    # with CLI flags as overrides. Drive the real run_pymc_recovery (only the
    # sampler itself is faked) and assert the block's values reach it.
    mod = _load_script("pymc_recover.py")

    stimuli = tmp_path / "stimuli.json"
    stimuli.write_text(
        '[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "model_module: src.subjective_randomness.model_families.prototype_similarity\n"
        f"stimuli_path: {stimuli}\n"
        "simulation: {n_participants: 2, n_repeats: 1, seed: 7}\n"
        "mcmc: {draws: 12, tune: 6, chains: 3, cores: 2}\n",
        encoding="utf-8",
    )

    class _Param:
        values = [0.5, 0.6]

    class _Idata:
        posterior = {
            name: _Param() for name in ("theta_alt", "alt_weight", "beta", "side_bias")
        }

    class _Fitted:
        fingerprint = "fake-fit"
        idata = _Idata()

    captured = {}

    def fake_fit_model(name, models_dir, responses_path, **kwargs):
        captured.update(kwargs)
        return _Fitted()

    monkeypatch.setattr("src.models.pymc_inference.fit_model", fake_fit_model)

    mod.main(mod.Args(config=config, out=tmp_path / "r.json"))

    assert captured["draws"] == 12
    assert captured["tune"] == 6
    assert captured["chains"] == 3
    assert captured["cores"] == 2


def test_model_recovery_writes_tidy_csv_when_flag_set(tmp_path, monkeypatch):
    mod = _load_script("model_recovery.py")
    monkeypatch.setattr(mod, "run_recovery_from_config", lambda *a, **k: CONFUSION_RESULT)
    tidy = tmp_path / "confusion.csv"

    mod.main(
        mod.Args(config=_write_config(tmp_path), out=tmp_path / "c.json", tidy_csv=tidy)
    )

    assert tidy.exists()
    rows = list(csv.DictReader(tidy.open(encoding="utf-8")))
    assert set(rows[0]) == CONFUSION_TIDY_COLUMNS
    assert len(rows) == 2  # 1 generating model x 2 recovered (seed) models


# ── analyze_recovery output flags execute (no monkeypatch: analysis is pure) ──


def _write_confusion_results(tmp_path: Path) -> Path:
    results = tmp_path / "confusion.json"
    results.write_text(json.dumps(CONFUSION_RESULT), encoding="utf-8")
    return results


def test_analyze_recovery_writes_out_csv_when_flag_set(tmp_path):
    mod = _load_script("analyze_recovery.py")
    out_csv = tmp_path / "summary.csv"

    mod.main(mod.Args(results=_write_confusion_results(tmp_path), out_csv=out_csv))

    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert set(rows[0]) == {
        "generating_model",
        "true_posterior",
        "best_by_posterior",
        "best_by_elpd",
        "correct_posterior",
        "correct_elpd",
        "winner_by_elpd",
        "winner_margin",
        "winner_margin_dse",
        "winner_distinguishable",
        "true_model_elpd_diff",
        "true_model_dse",
        "recovery_clear",
    }


def test_analyze_recovery_writes_figure_when_flag_set(tmp_path):
    mod = _load_script("analyze_recovery.py")
    figure = tmp_path / "confusion.png"

    mod.main(mod.Args(results=_write_confusion_results(tmp_path), figure=figure))

    assert figure.exists() and figure.stat().st_size > 0


def test_analyze_recovery_skips_outputs_when_flags_absent(tmp_path):
    mod = _load_script("analyze_recovery.py")

    mod.main(mod.Args(results=_write_confusion_results(tmp_path)))

    assert list(tmp_path.glob("*.csv")) == []  # no summary CSV
    assert list(tmp_path.glob("*.png")) == []  # no figure


# A confusion carrying the comparison table, so the analysis CLI exercises its
# has_comparison print branch and populates the distinguishability CSV columns.
CONFUSION_WITH_COMPARISON = {
    "seed_models": ["A", "B"],
    "generator": "pymc",
    "generating": [
        {  # recovered AND clearly ahead (margin 20 > 2*3)
            "generating_model": "A",
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.9, "B": 0.1},
            "elpd_loo": {"A": -10.0, "B": -30.0},
            "comparison": {
                "A": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "weight": 0.9},
                "B": {"rank": 1, "elpd_diff": 20.0, "dse": 3.0, "weight": 0.1},
            },
        },
        {  # mis-recovered AND tied (margin 0.5 < 2*1.2)
            "generating_model": "B",
            "best_model": "A",
            "recovered_correct": False,
            "posteriors": {"A": 0.55, "B": 0.45},
            "elpd_loo": {"A": -9.0, "B": -9.5},
            "comparison": {
                "A": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "weight": 0.55},
                "B": {"rank": 1, "elpd_diff": 0.5, "dse": 1.2, "weight": 0.45},
            },
        },
    ],
}


def test_analyze_recovery_csv_populates_distinguishability_with_comparison(tmp_path):
    mod = _load_script("analyze_recovery.py")
    results = tmp_path / "conf.json"
    results.write_text(json.dumps(CONFUSION_WITH_COMPARISON), encoding="utf-8")
    out_csv = tmp_path / "summary.csv"

    mod.main(mod.Args(results=results, out_csv=out_csv))

    by_model = {
        r["generating_model"]: r
        for r in csv.DictReader(out_csv.open(encoding="utf-8"))
    }
    # CSV booleans are written as "True"/"False" (not blank as in the no-comparison case).
    assert by_model["A"]["winner_distinguishable"] == "True"
    assert by_model["A"]["recovery_clear"] == "True"
    assert by_model["B"]["winner_distinguishable"] == "False"
    assert by_model["B"]["recovery_clear"] == "False"


# A sampled-truth parameter-recovery report (pymc_recover.py's default mode):
# no top-level true_params; each run pairs its own ground truth with a
# posterior summary.
def _posterior(**means: float) -> dict:
    return {
        name: {"mean": mean, "q025": mean - 0.3, "q975": mean + 0.3}
        for name, mean in means.items()
    }


SAMPLED_PARAM_REPORT = {
    "model": "demo",
    "n_repeats": 4,
    "n_stimuli": 4,
    "param_ranges": {"theta": [0.0, 1.0], "beta": [0.2, 12.0]},
    "runs": [
        {
            "repeat": 0,
            "true_params": {"theta": 0.1, "beta": 1.0},
            "posterior": _posterior(theta=0.15, beta=1.4),
        },
        {
            "repeat": 1,
            "true_params": {"theta": 0.4, "beta": 4.0},
            "posterior": _posterior(theta=0.35, beta=3.2),
        },
        {
            "repeat": 2,
            "true_params": {"theta": 0.6, "beta": 7.0},
            "posterior": _posterior(theta=0.7, beta=8.1),
        },
        {
            "repeat": 3,
            "true_params": {"theta": 0.9, "beta": 10.0},
            "posterior": _posterior(theta=0.85, beta=11.0),
        },
    ],
}


def _write_results(tmp_path: Path, payload: dict) -> Path:
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload), encoding="utf-8")
    return results


def test_analyze_recovery_sampled_report_csv_includes_pearson_r(tmp_path):
    mod = _load_script("analyze_recovery.py")
    out_csv = tmp_path / "summary.csv"

    mod.main(
        mod.Args(results=_write_results(tmp_path, SAMPLED_PARAM_REPORT), out_csv=out_csv)
    )

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert "pearson_r" in rows[0]
    by_param = {r["parameter"]: r for r in rows}
    # Truths and estimates rise together, so the correlations are strong.
    assert float(by_param["theta"]["pearson_r"]) > 0.9
    assert float(by_param["beta"]["pearson_r"]) > 0.9
    # No single true value exists when truths vary across repeats.
    assert by_param["theta"]["true_value"] == ""


def test_analyze_recovery_sampled_report_writes_correlation_figure(tmp_path):
    mod = _load_script("analyze_recovery.py")
    figure = tmp_path / "recovery.png"

    mod.main(
        mod.Args(results=_write_results(tmp_path, SAMPLED_PARAM_REPORT), figure=figure)
    )

    assert figure.exists() and figure.stat().st_size > 0


def test_analyze_recovery_fixed_report_still_writes_figure(tmp_path):
    # The old fixed-truth shape (e.g. pymc_recover.py output) must keep working.
    mod = _load_script("analyze_recovery.py")
    figure = tmp_path / "fixed.png"

    mod.main(mod.Args(results=_write_results(tmp_path, PARAM_REPORT), figure=figure))

    assert figure.exists() and figure.stat().st_size > 0


# ── run_recovery_pipeline CLI ───────────────────────────────────────


def test_run_recovery_pipeline_cli_defaults_and_overrides():
    args_cls = _load_script("run_recovery_pipeline.py").Args

    default = tyro.cli(args_cls, args=[])
    assert default.out_dir == Path("data/subjective_randomness/recovery_pipeline")
    assert len(default.param_configs) == 3  # one per model family
    assert all(p.suffix == ".yaml" for p in default.param_configs)
    assert default.model_recovery_config.name == "model_recovery.yaml"
    assert default.skip_model_recovery is False
    assert default.selection_comparison_config.name == "selection_comparison.yaml"
    assert default.skip_selection_comparison is False
    assert default.n_repeats is None
    assert default.draws is None

    full = tyro.cli(
        args_cls,
        args=[
            "--out-dir", "o",
            "--param-configs", "a.yaml", "b.yaml",
            "--skip-model-recovery",
            "--skip-selection-comparison",
            "--n-repeats", "5",
            "--draws", "100",
        ],
    )
    assert full.out_dir == Path("o")
    assert full.param_configs == (Path("a.yaml"), Path("b.yaml"))
    assert full.skip_model_recovery is True
    assert full.skip_selection_comparison is True
    assert full.n_repeats == 5
    assert full.draws == 100


def test_run_recovery_pipeline_main_passes_none_configs_when_skipping(monkeypatch, tmp_path):
    mod = _load_script("run_recovery_pipeline.py")
    seen = {}

    def fake_run_pipeline(param_configs, out_dir, model_config, **kwargs):
        seen["model_config"] = model_config
        seen["selection_comparison_config"] = kwargs["selection_comparison_config_path"]
        key_results = tmp_path / "key_results.txt"
        key_results.write_text("key results", encoding="utf-8")
        return {"reports": [], "confusion": None, "key_results_path": key_results}

    monkeypatch.setattr(mod, "run_pipeline", fake_run_pipeline)

    mod.main(
        mod.Args(
            out_dir=tmp_path, skip_model_recovery=True, skip_selection_comparison=True
        )
    )
    assert seen["model_config"] is None
    assert seen["selection_comparison_config"] is None

    mod.main(mod.Args(out_dir=tmp_path))
    assert seen["model_config"] is not None
    assert seen["selection_comparison_config"].name == "selection_comparison.yaml"


# ── select_stimuli CLI ──────────────────────────────────────────────


def test_select_stimuli_cli_defaults_and_paths():
    args_cls = _load_script("select_stimuli.py").Args

    default = tyro.cli(args_cls, args=["--candidates", "c.json", "--out", "o.json"])
    assert default.candidates == Path("c.json")
    assert default.out == Path("o.json")
    assert default.top is None
    assert default.models is None
    assert default.param_samples is None


def test_select_stimuli_writes_ranked_top_k(tmp_path):
    mod = _load_script("select_stimuli.py")
    candidates = tmp_path / "pool.json"
    candidates.write_text(
        json.dumps(
            [
                {"sequence_a": "HHHHTTTT", "sequence_b": "HTHTHTHT"},
                {"sequence_a": "HHHHHHHH", "sequence_b": "HTTHTHHT"},
                {"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT"},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "selected.json"

    mod.main(mod.Args(candidates=candidates, out=out, top=2))

    assert out.exists()
    selected = json.loads(out.read_text(encoding="utf-8"))
    assert len(selected) == 2  # top-2 kept
    eigs = [s["discrimination_eig"] for s in selected]
    assert eigs == sorted(eigs, reverse=True)  # ranked descending
    assert all({"sequence_a", "sequence_b"} <= set(s) for s in selected)


# ── adaptive_recover CLI ────────────────────────────────────────────


def _write_yaml(tmp_path: Path, name: str, config: dict) -> Path:
    # JSON is valid YAML, so load_config (yaml.safe_load) reads this fine.
    path = tmp_path / name
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_adaptive_recover_cli_defaults():
    args_cls = _load_script("adaptive_recover.py").Args
    default = tyro.cli(args_cls, args=["--config", "c.yaml", "--out", "o.json"])
    assert default.mode is None
    assert default.selected_out is None
    assert default.n_rounds is None
    assert default.seed is None


def test_adaptive_recover_parameter_mode_writes_report_and_selected(tmp_path):
    mod = _load_script("adaptive_recover.py")
    config = _write_yaml(
        tmp_path,
        "param.yaml",
        {
            "mode": "parameter",
            "model": "prototype_similarity",
            "true_params": None,
            "pool": {"n_pairs": 30, "lengths": [6]},
            "n_rounds": 8,
            "n_participants": 20,
            "points_per_dim": 4,
            "seed": 0,
        },
    )
    out = tmp_path / "param_report.json"
    selected = tmp_path / "design.json"

    mod.main(mod.Args(config=config, out=out, selected_out=selected))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["model"] == "prototype_similarity"
    assert set(report["posterior_mean"]) == set(report["true_params"])
    assert len(report["selected_stimuli"]) == 8
    assert report["final_entropy_bits"] < report["prior_entropy_bits"]
    # The design (selected stimuli) was written out separately.
    assert len(json.loads(selected.read_text(encoding="utf-8"))) == 8


def test_adaptive_recover_model_mode_writes_confusion(tmp_path):
    mod = _load_script("adaptive_recover.py")
    config = _write_yaml(
        tmp_path,
        "model.yaml",
        {
            "mode": "model",
            "model_names": [
                "prototype_similarity",
                "encoding_compressibility",
                "bayesian_diagnosticity",
            ],
            "generating_models": {"encoding_compressibility": None},
            "pool": {"n_pairs": 30, "lengths": [6]},
            "n_rounds": 8,
            "n_participants": 30,
            "points_per_dim": 4,
            "seed": 0,
        },
    )
    out = tmp_path / "model_report.json"

    mod.main(mod.Args(config=config, out=out))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert [e["generating_model"] for e in report["generating"]] == [
        "encoding_compressibility"
    ]
    entry = report["generating"][0]
    assert entry["recovered_model"] in report["model_names"]
    assert abs(sum(entry["model_posterior"].values()) - 1.0) < 1e-9
