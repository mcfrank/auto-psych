"""Run the whole subjective-randomness recovery pipeline in one call.

Stages:

1. *Parameter recovery* (`pymc_recover.run_pymc_recovery`) for each
   model-family config: simulate from the reference family, fit the matching
   PyMC adapter (parameter fitting is Bayesian-only in this pipeline), and
   write the report JSON, tidy CSV, per-parameter summary CSV, and the
   recovery figure (ground-truth vs. recovered correlation scatters for the
   default sampled-truth configs).
2. *Stimulus-selection comparison* (`adaptive_recovery.compare_*`): the same
   grid-posterior recovery run twice per sampled ground truth — once on the
   top-EIG stimulus set, once on a random same-size set from the same pool —
   for both parameter and model recovery, so the arms differ only in which
   stimulus set was chosen. Optional.
3. *Closed-ended model recovery* (`model_recovery.run_recovery_from_config`):
   confusion JSON, tidy CSV, and the posterior confusion heatmap. Optional,
   because it fits every seed model with MCMC and is by far the slowest stage.
4. `key_results.txt` aggregating the per-stage text summaries.

Every artifact lands in one output directory, so a pipeline run is a single
self-contained bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.subjective_randomness.analysis import parameter_recovery_summary
from src.subjective_randomness.config import load_config
from src.subjective_randomness.pymc_recover import run_pymc_recovery
from src.subjective_randomness.reporting import (
    PARAM_SUMMARY_COLUMNS,
    model_recovery_text,
    parameter_recovery_text,
    plot_model_recovery,
    plot_parameter_recovery,
    plot_selection_comparison_models,
    plot_selection_comparison_parameters,
    selection_comparison_model_text,
    selection_comparison_parameter_text,
)
from src.subjective_randomness.tidy import parameter_recovery_tidy_rows, write_tidy_csv


def run_parameter_recovery_stage(
    config_path: Path,
    out_dir: Path,
    n_repeats: Optional[int] = None,
    mcmc_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one family's Bayesian (PyMC) parameter recovery, write artifacts.

    Writes `{model}_recovery.json`, `{model}_recovery_tidy.csv`,
    `{model}_summary.csv`, and `{model}_correlation.png` into `out_dir`.
    """
    config = load_config(config_path)
    # run_pymc_recovery resolves the config's `mcmc` block itself; only the
    # explicit CLI overrides are forwarded.
    report = run_pymc_recovery(
        config,
        config_path,
        repeats_override=n_repeats,
        **{k: v for k, v in (mcmc_overrides or {}).items() if v is not None},
    )
    model = report["model"]
    report_path = out_dir / f"{model}_recovery.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_tidy_csv(
        parameter_recovery_tidy_rows(report), out_dir / f"{model}_recovery_tidy.csv"
    )
    write_tidy_csv(
        parameter_recovery_summary(report),
        out_dir / f"{model}_summary.csv",
        columns=PARAM_SUMMARY_COLUMNS,
    )
    plot_parameter_recovery(report, out_dir / f"{model}_correlation.png")
    return report


def run_selection_comparison_stage(
    config_path: Path, out_dir: Path
) -> Dict[str, Any]:
    """Run the EIG-optimized vs. random stimulus-set comparison stage.

    Writes `selection_comparison_{model}.json` and `.png` per family plus
    `selection_comparison_model_recovery.json` / `.png` into `out_dir`.
    """
    from src.subjective_randomness.adaptive_recovery import (
        compare_model_recovery,
        compare_parameter_recovery,
    )
    from src.subjective_randomness.stimulus_design import generate_candidate_pool

    config = load_config(config_path)
    model_names = list(config["model_names"])
    seed = int(config.get("seed", 0))
    pool_config = dict(config.get("pool", {}))
    pool = generate_candidate_pool(
        n_pairs=int(pool_config.get("n_pairs", 200)),
        lengths=tuple(pool_config.get("lengths", (6, 8))),
        seed=seed,
    )
    # Parameter and model-identity recovery carry very different information
    # per response, so each comparison gets its own data budget.
    shared = dict(
        n_stimuli=int(config["n_stimuli"]),
        points_per_dim=int(config.get("points_per_dim", 7)),
        seed=seed,
    )

    parameter_reports: List[Dict[str, Any]] = []
    for model_name in model_names:
        report = compare_parameter_recovery(
            model_name,
            pool,
            n_repeats=int(config["parameter_repeats"]),
            n_participants=int(config["parameter_participants"]),
            **shared,
        )
        (out_dir / f"selection_comparison_{model_name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        plot_selection_comparison_parameters(
            report, out_dir / f"selection_comparison_{model_name}.png"
        )
        parameter_reports.append(report)
        print(f"selection comparison done: {model_name}")

    model_report = compare_model_recovery(
        pool,
        model_names=model_names,
        n_repeats=int(config["model_repeats"]),
        n_participants=int(config["model_participants"]),
        **shared,
    )
    (out_dir / "selection_comparison_model_recovery.json").write_text(
        json.dumps(model_report, indent=2), encoding="utf-8"
    )
    plot_selection_comparison_models(
        model_report, out_dir / "selection_comparison_model_recovery.png"
    )
    print("selection comparison done: model recovery")
    return {"parameter": parameter_reports, "model": model_report}


def run_model_recovery_stage(
    config_path: Path,
    out_dir: Path,
    *,
    n_participants: Optional[int] = None,
    fit_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run closed-ended model recovery and write its artifacts.

    Writes `confusion.json`, `confusion.csv`, and `confusion.png` into
    `out_dir`; per-model inner-loop artifacts go to `model_recovery_runs/`.
    """
    # Imported lazily: this pulls PyMC, which parameter-recovery-only runs
    # never need.
    from src.subjective_randomness.model_recovery import (
        CONFUSION_COLUMNS,
        confusion_tidy_rows,
        run_recovery_from_config,
    )

    confusion = run_recovery_from_config(
        load_config(config_path),
        config_path,
        out_dir / "model_recovery_runs",
        n_participants_override=n_participants,
        fit_overrides=fit_overrides,
    )
    (out_dir / "confusion.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )
    write_tidy_csv(
        confusion_tidy_rows(confusion),
        out_dir / "confusion.csv",
        columns=CONFUSION_COLUMNS,
    )
    plot_model_recovery(confusion, out_dir / "confusion.png")
    return confusion


def key_results_text(
    reports: Sequence[Mapping[str, Any]],
    confusion: Optional[Mapping[str, Any]],
    selection_comparison: Optional[Mapping[str, Any]] = None,
) -> str:
    """One text document with every stage's summary table."""
    sections = ["Subjective-randomness recovery pipeline — key results", ""]
    for report in reports:
        sections.append(parameter_recovery_text(report))
        sections.append("")
    if confusion is None:
        sections.append("Closed-ended model recovery: skipped.")
    else:
        sections.append(model_recovery_text(confusion))
    sections.append("")
    if selection_comparison is None:
        sections.append("Stimulus-selection comparison: skipped.")
    else:
        for report in selection_comparison["parameter"]:
            sections.append(selection_comparison_parameter_text(report))
            sections.append("")
        sections.append(selection_comparison_model_text(selection_comparison["model"]))
    return "\n".join(sections) + "\n"


def run_pipeline(
    param_config_paths: Sequence[Path],
    out_dir: Path,
    model_recovery_config_path: Optional[Path],
    *,
    selection_comparison_config_path: Optional[Path] = None,
    n_repeats: Optional[int] = None,
    n_participants: Optional[int] = None,
    fit_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every recovery stage and aggregate the key results.

    `model_recovery_config_path=None` skips the model-recovery stage and
    `selection_comparison_config_path=None` skips the EIG-vs-random
    stimulus-set comparison. `n_repeats` overrides each parameter-recovery
    config's repeat count; `fit_overrides` (draws/tune/chains) tunes MCMC in
    both the parameter-recovery and model-recovery stages, and
    `n_participants` the model-recovery stage. Returns the parameter-recovery
    reports, the confusion result and selection comparison (each ``None`` when
    skipped), and the path of the key-results text file.
    """
    if not param_config_paths:
        raise ValueError(
            "No parameter-recovery configs given; the pipeline needs at least one."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for config_path in param_config_paths:
        report = run_parameter_recovery_stage(
            Path(config_path), out_dir, n_repeats, fit_overrides
        )
        print(
            f"parameter recovery done: {report['model']} "
            f"({report['n_repeats']} repeats, {report['n_stimuli']} stimuli)"
        )
        reports.append(report)

    selection_comparison = None
    if selection_comparison_config_path is not None:
        selection_comparison = run_selection_comparison_stage(
            Path(selection_comparison_config_path), out_dir
        )

    # The MCMC-heavy stage runs last so the faster stages' artifacts are on
    # disk even if it fails.
    confusion = None
    if model_recovery_config_path is not None:
        confusion = run_model_recovery_stage(
            Path(model_recovery_config_path),
            out_dir,
            n_participants=n_participants,
            fit_overrides=fit_overrides,
        )
        print(f"model recovery done: {len(confusion['generating'])} generating models")

    key_results_path = out_dir / "key_results.txt"
    key_results_path.write_text(
        key_results_text(reports, confusion, selection_comparison), encoding="utf-8"
    )
    return {
        "reports": reports,
        "confusion": confusion,
        "selection_comparison": selection_comparison,
        "key_results_path": key_results_path,
    }
