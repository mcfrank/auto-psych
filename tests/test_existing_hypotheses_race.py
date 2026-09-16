"""Tests for the race presentation in existing_hypotheses.md.

The rewritten ``_write_existing_hypotheses`` should show candidates the
``az.compare`` race (rank, ``elpd_diff ± dse``, a verdict, and PSIS-LOO
reliability) rather than the rounded softmax posterior.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml

from src.pipelines.inner_loop.pymc_orchestrator import (
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    _write_existing_hypotheses,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_models_dir(tmp_path: Path, names_and_rationales: dict) -> Path:
    """Create a models dir with a manifest."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    entries = [
        {"name": name, "rationale": rationale}
        for name, rationale in names_and_rationales.items()
    ]
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": entries}, sort_keys=False), encoding="utf-8"
    )
    return models_dir


def _make_comparison(
    models: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a comparison table dict (same shape as ``compare_table`` output)."""
    return models


def _make_posterior(
    posteriors: dict[str, float],
    elpd_loo: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    return {
        "posteriors": posteriors,
        "elpd_loo": elpd_loo or {},
    }


def _make_candidate_dir(tmp_path: Path) -> Path:
    d = tmp_path / "candidate_0"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── tests ────────────────────────────────────────────────────────────────


class TestRacePresentation:
    """With a comparison table, existing_hypotheses.md shows the race."""

    def test_best_first_ordering(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {
            "alpha": "Alpha hypothesis",
            "beta": "Beta hypothesis",
            "gamma": "Gamma hypothesis",
        })
        comparison = {
            "alpha": {"rank": 1, "elpd_loo": -50.0, "elpd_diff": 3.2, "dse": 2.0,
                       "weight": 0.3, "loo_unreliable": False},
            "beta":  {"rank": 0, "elpd_loo": -46.8, "elpd_diff": 0.0, "dse": 0.0,
                       "weight": 0.5, "loo_unreliable": False},
            "gamma": {"rank": 2, "elpd_loo": -60.0, "elpd_diff": 13.2, "dse": 3.0,
                       "weight": 0.2, "loo_unreliable": False},
        }
        posterior = _make_posterior({"alpha": 0.3, "beta": 0.5, "gamma": 0.2})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        lines = text.split("\n")
        model_headers = [l for l in lines if l.startswith("## ")]
        assert model_headers[0].startswith("## beta")
        assert model_headers[1].startswith("## alpha")
        assert model_headers[2].startswith("## gamma")

    def test_rank_and_elpd_diff_shown(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {
            "best": "The best model",
            "runner": "Second place",
        })
        comparison = {
            "best":   {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                        "weight": 0.7, "loo_unreliable": False},
            "runner": {"rank": 1, "elpd_loo": -43.5, "elpd_diff": 3.5, "dse": 2.1,
                        "weight": 0.3, "loo_unreliable": False},
        }
        posterior = _make_posterior({"best": 0.7, "runner": 0.3})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "rank 0" in text
        assert "rank 1" in text
        assert "3.5" in text
        assert "2.1" in text

    def test_tied_verdict(self, tmp_path):
        """A model within 2·dse of the best is 'not clearly separated'."""
        models_dir = _make_models_dir(tmp_path, {
            "best": "Best",
            "close": "Close rival",
        })
        comparison = {
            "best":  {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                       "weight": 0.7, "loo_unreliable": False},
            "close": {"rank": 1, "elpd_loo": -42.0, "elpd_diff": 2.0, "dse": 2.0,
                       "weight": 0.3, "loo_unreliable": False},
        }
        posterior = _make_posterior({"best": 0.7, "close": 0.3})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        # The close model's verdict should indicate it's still in the race
        close_section = text.split("## close")[1].split("## ")[0] if "## close" in text else ""
        assert "tied" in close_section.lower() or "not clearly separated" in close_section.lower()

    def test_loser_verdict(self, tmp_path):
        """A model beyond 2·dse of the best is 'clearly behind'."""
        models_dir = _make_models_dir(tmp_path, {
            "best": "Best",
            "loser": "Far behind",
        })
        comparison = {
            "best":  {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                       "weight": 0.7, "loo_unreliable": False},
            "loser": {"rank": 1, "elpd_loo": -55.0, "elpd_diff": 15.0, "dse": 3.0,
                       "weight": 0.3, "loo_unreliable": False},
        }
        posterior = _make_posterior({"best": 0.7, "loser": 0.3})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        loser_section = text.split("## loser")[1].split("## ")[0] if "## loser" in text else ""
        assert "lost" in loser_section.lower() or "clearly behind" in loser_section.lower()

    def test_unreliable_loo_shown(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {
            "good": "Good model",
            "shaky": "Model with unreliable LOO",
        })
        comparison = {
            "good":  {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                       "weight": 0.7, "loo_unreliable": False},
            "shaky": {"rank": 1, "elpd_loo": -42.0, "elpd_diff": 2.0, "dse": 1.5,
                       "weight": 0.3, "loo_unreliable": True, "frac_bad_k": 0.25},
        }
        posterior = _make_posterior({"good": 0.7, "shaky": 0.3})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "unreliable" in text.lower()
        assert "25%" in text

    def test_no_softmax_posterior(self, tmp_path):
        """The text must not contain softmax posterior values."""
        models_dir = _make_models_dir(tmp_path, {
            "a": "Model A",
            "b": "Model B",
        })
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 0.7, "loo_unreliable": False},
            "b": {"rank": 1, "elpd_loo": -43.0, "elpd_diff": 3.0, "dse": 2.0,
                   "weight": 0.3, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 0.97, "b": 0.03})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "posterior 0.97" not in text.lower()
        assert "posterior 0.03" not in text.lower()
        assert "0.970" not in text
        assert "0.030" not in text

    def test_no_per_stimulus_residuals(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "residual" not in text.lower()

    def test_no_parameter_values(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "parameter" not in text.lower()

    def test_no_source_code(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "import " not in text
        assert "pm.Model" not in text
        assert "def " not in text

    def test_best_model_rank_zero(self, tmp_path):
        """The best model (rank 0) shows as rank 0, best position."""
        models_dir = _make_models_dir(tmp_path, {"best": "The best"})
        comparison = {
            "best": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                      "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"best": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "rank 0" in text

    def test_file_written(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        written = (candidate_dir / "existing_hypotheses.md").read_text(encoding="utf-8")
        assert "Model A" in written


class TestWithoutComparison:
    """Without a comparison table (first scoring, no data yet), fall back."""

    def test_fallback_shows_elpd_if_available(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        posterior = _make_posterior({"a": 0.5}, elpd_loo={"a": -42.5})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(candidate_dir, models_dir, posterior)

        assert "-42.5" in text

    def test_fallback_manifest_order(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {
            "zebra": "Z model",
            "alpha": "A model",
        })
        posterior = _make_posterior({"zebra": 0.5, "alpha": 0.5})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(candidate_dir, models_dir, posterior)

        z_pos = text.index("## zebra")
        a_pos = text.index("## alpha")
        assert z_pos < a_pos  # manifest order preserved


class TestModelWithNoComparisonRow:
    """A model present in the manifest but absent from comparison should not crash."""

    def test_no_crash_on_missing_comparison_row(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {
            "a": "Model A",
            "new_model": "Just admitted, not yet scored",
        })
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert "new_model" in text
        assert "no comparison row" in text.lower()


class TestDescribeStanding:
    """The _describe_standing helper should produce correct one-line summaries."""

    def test_best_model(self):
        from src.pipelines.inner_loop.pymc_orchestrator import _describe_standing

        row = {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
               "loo_unreliable": False}
        text = _describe_standing(row)
        assert "rank 0" in text
        assert "best" in text.lower()

    def test_tied_model(self):
        from src.pipelines.inner_loop.pymc_orchestrator import _describe_standing

        row = {"rank": 1, "elpd_loo": -42.0, "elpd_diff": 2.0, "dse": 2.0,
               "loo_unreliable": False}
        text = _describe_standing(row)
        assert "rank 1" in text
        assert "tied" in text.lower()

    def test_losing_model(self):
        from src.pipelines.inner_loop.pymc_orchestrator import _describe_standing

        row = {"rank": 2, "elpd_loo": -55.0, "elpd_diff": 15.0, "dse": 3.0,
               "loo_unreliable": False}
        text = _describe_standing(row)
        assert "rank 2" in text
        assert "lost" in text.lower()

    def test_unreliable_loo(self):
        from src.pipelines.inner_loop.pymc_orchestrator import _describe_standing

        row = {"rank": 1, "elpd_loo": -45.0, "elpd_diff": 5.0, "dse": 3.0,
               "loo_unreliable": True, "frac_bad_k": 0.15}
        text = _describe_standing(row)
        assert "unreliable" in text.lower()
        assert "15%" in text


class TestDseMultiplierInPreamble:
    """The preamble text should reference the actual dse multiplier."""

    def test_preamble_mentions_dse_threshold(self, tmp_path):
        models_dir = _make_models_dir(tmp_path, {"a": "Model A"})
        comparison = {
            "a": {"rank": 0, "elpd_loo": -40.0, "elpd_diff": 0.0, "dse": 0.0,
                   "weight": 1.0, "loo_unreliable": False},
        }
        posterior = _make_posterior({"a": 1.0})
        candidate_dir = _make_candidate_dir(tmp_path)

        text = _write_existing_hypotheses(
            candidate_dir, models_dir, posterior, comparison=comparison
        )

        assert f"{DEFAULT_PRUNE_DSE_MULTIPLIER:g}" in text
        assert "dse" in text.lower()
