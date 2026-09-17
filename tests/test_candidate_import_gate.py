"""Import gate: candidates and critique statistics with forbidden imports are rejected."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipelines.inner_loop.import_gate import (
    CANDIDATE_IMPORT_ALLOWLIST,
    check_forbidden_imports,
)


class TestCheckForbiddenImports:
    """Pure function: AST-level import scanning."""

    def test_allowed_imports_pass(self):
        source = textwrap.dedent("""\
            import numpy as np
            import pymc as pm
            import pytensor.tensor as pt
            from scipy.stats import norm
            from math import log
            from typing import Dict
            from collections import Counter
            import re
            import itertools
            import functools
            import dataclasses
            import statistics
            import operator
            import arviz
        """)
        assert check_forbidden_imports(source) == []

    def test_forbidden_top_level_import(self):
        source = "import pandas as pd\n"
        result = check_forbidden_imports(source)
        assert "pandas" in result

    def test_forbidden_from_import(self):
        source = "from src.subjective_randomness.features import featurize_stimulus\n"
        result = check_forbidden_imports(source)
        assert any("src" in r for r in result)

    def test_forbidden_import_inside_function(self):
        source = textwrap.dedent("""\
            import numpy as np
            def compute_features(seq_a, seq_b):
                from src.subjective_randomness.features import parse_motifs
                return {"motif": parse_motifs(seq_a)}
        """)
        result = check_forbidden_imports(source)
        assert len(result) > 0
        assert any("src" in r for r in result)

    def test_relative_import_forbidden(self):
        source = "from . import something\n"
        result = check_forbidden_imports(source)
        assert len(result) > 0

    def test_unparseable_source(self):
        source = "def broken(\n"
        result = check_forbidden_imports(source)
        assert len(result) > 0

    def test_empty_source_passes(self):
        assert check_forbidden_imports("") == []

    def test_nested_submodule_allowed(self):
        source = "from scipy.special import expit\n"
        assert check_forbidden_imports(source) == []


class TestAdmitCandidateImportGate:
    """_admit_candidate rejects forbidden imports before any other check."""

    @pytest.fixture()
    def setup(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        responses_path = tmp_path / "responses.csv"
        responses_path.write_text(
            "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
            "HHT,THT,0,0,1\n",
            encoding="utf-8",
        )
        return models_dir, responses_path

    def test_rejects_candidate_with_forbidden_import(self, tmp_path, setup):
        from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
        from src.pipelines.inner_loop.pymc_orchestrator import _admit_candidate

        models_dir, responses_path = setup
        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()

        (candidate_dir / "candidate.py").write_text(
            textwrap.dedent("""\
                import pandas as pd
                import pymc as pm
                import numpy as np
                with pm.Model() as model:
                    pass
            """),
            encoding="utf-8",
        )
        (candidate_dir / "hypothesis.md").write_text(
            "Test hypothesis.\n", encoding="utf-8"
        )
        ledger = HypothesisLedger.create(
            tmp_path / "ledger.jsonl", inherit_from=None
        )

        result = _admit_candidate(
            candidate_dir / "candidate.py",
            models_dir,
            "test_model",
            responses_path,
            ledger=ledger,
            ledger_context="test",
        )
        assert result is False

        entries = ledger.entries()
        assert len(entries) == 1
        assert entries[0].outcome == "rejected"
        assert "forbidden import" in entries[0].detail

    def test_admits_candidate_with_allowed_imports(self, tmp_path, setup):
        from src.pipelines.inner_loop.pymc_orchestrator import _admit_candidate

        models_dir, responses_path = setup
        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()
        (candidate_dir / "candidate.py").write_text(
            textwrap.dedent("""\
                import numpy as np
                import pymc as pm
                import pytensor.tensor as pt
                with pm.Model() as model:
                    x = pm.Data("chose_left", np.zeros(1, dtype="int64"))
                    p = pm.Beta("p", 1, 1)
                    pm.Bernoulli("response", p=p, observed=x)
            """),
            encoding="utf-8",
        )
        (candidate_dir / "hypothesis.md").write_text(
            "Test hypothesis.\n", encoding="utf-8"
        )

        # The model will fail later gates (logp etc.) but should pass the import gate.
        # We just check it doesn't get rejected for imports by patching past the later gates.
        with patch(
            "src.pipelines.inner_loop.pymc_orchestrator.load_pymc_model"
        ) as mock_load, patch(
            "src.pipelines.inner_loop.pymc_orchestrator.model_logp_is_finite",
            return_value=(True, ""),
        ), patch(
            "src.pipelines.inner_loop.pymc_orchestrator.fit_model"
        ), patch(
            "src.pipelines.inner_loop.pymc_orchestrator.log_likelihood",
            return_value=-10.0,
        ), patch(
            "src.pipelines.inner_loop.pymc_orchestrator._min_prediction_rmse",
            return_value=(None, float("inf")),
        ):
            result = _admit_candidate(
                candidate_dir / "candidate.py",
                models_dir,
                "test_model",
                responses_path,
            )
            assert result is True


class TestCritiqueImportGate:
    """Critique test_statistic files with forbidden imports are filtered out."""

    def test_forbidden_import_in_test_statistic(self):
        source = textwrap.dedent("""\
            # name: bad_stat
            # description: uses pandas
            import pandas as pd
            def test_statistic(df):
                return float(df["chose_left"].mean())
        """)
        result = check_forbidden_imports(source)
        assert "pandas" in result

    def test_allowed_test_statistic(self):
        source = textwrap.dedent("""\
            # name: good_stat
            # description: uses only numpy
            import numpy as np
            def test_statistic(df):
                return float(np.mean(df["chose_left"]))
        """)
        assert check_forbidden_imports(source) == []


class TestBriefsStateAllowlist:
    """The candidate and critique briefs name the import allowlist."""

    def test_candidate_context_states_allowlist(self, tmp_path):
        from src.pipelines.inner_loop.pymc_orchestrator import _write_candidate_context

        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()
        responses_path = tmp_path / "responses.csv"
        responses_path.write_text(
            "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
            "HHT,THT,0,0,1\n",
            encoding="utf-8",
        )
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        docs = _write_candidate_context(
            candidate_dir,
            responses_path,
            models_dir,
            iteration=0,
            candidate_idx=0,
            candidate_count=1,
            current_posterior=None,
        )
        context = docs["context"]
        assert "numpy" in context
        assert "pymc" in context
        assert "pytensor" in context
        assert "allowlist" in context.lower() or "allowed imports" in context.lower()
        assert "written in the file itself" in context.lower() or "self-contained" in context.lower()

    def test_pymc_theory_states_allowlist(self):
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "pipelines"
            / "inner_loop"
            / "prompts"
            / "pymc_theory.md"
        )
        text = prompt_path.read_text(encoding="utf-8")
        assert "numpy" in text
        assert "pymc" in text
        assert "pytensor" in text
        assert "allowlist" in text.lower() or "allowed imports" in text.lower()
