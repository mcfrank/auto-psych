"""The code gate for agent-written memo model files."""

import pytest

from src.pipelines.inner_loop.import_gate import check_forbidden_imports
from src.rsa.loop.code_gate import code_problems
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@pytest.mark.parametrize("path", sorted(SEEDS.glob("*.py")), ids=lambda p: p.stem)
def test_every_seed_passes(path):
    assert code_problems(path.read_text()) == []


@pytest.mark.parametrize(
    "source, problem",
    [
        ("import os", "os"),
        ("import src", "src"),
        ("from src.rsa import fit", "src.rsa"),
        ("from src.rsa.dataset import load_forced_choice", "src.rsa.dataset"),
        ("import pymc", "pymc"),
        ("import jax.numpy as jnp\nx = jnp.load('f.npy')", "use of .load"),
        ("x = open('f')", "use of open"),
        ("import numpyro\nnumpyro.sys", "use of .sys"),
    ],
)
def test_escapes_are_refused(source, problem):
    assert problem in code_problems(source)


def test_the_exact_module_does_not_open_its_package():
    assert code_problems("import src.rsa.memo_kit") == []
    assert code_problems("from src.rsa.memo_kit import at, vec") == []
    assert "src.rsa.memo_kit_extra" in code_problems("import src.rsa.memo_kit_extra")


def test_the_pymc_domain_default_is_unchanged():
    assert check_forbidden_imports("import pymc as pm") == []
    assert "jax" in check_forbidden_imports("import jax")
    assert "src.rsa.memo_kit" in check_forbidden_imports("from src.rsa.memo_kit import at")
