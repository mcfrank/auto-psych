"""Every seed model computes its own features from raw H/T sequences.

Guards:
* vendored helpers in each seed are byte-identical to the originals in
  ``src/subjective_randomness/model_families/common.py``;
* each seed's ``compute_features`` returns exactly the columns its ``pm.Data``
  containers read, with the correct values;
* no seed imports the project featurizer;
* the registry and live pool copies stay in step.
"""

from __future__ import annotations

import ast
from itertools import product

import pytest

from tests.paths import REPO_ROOT

REGISTRY = REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families"
POOL = (
    REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects"
    / "subjective_randomness" / "seed_models"
)
VENDORED = {
    "falk_konold_dp": (
        ("clean_sequence", "parse_motifs"),
        ("rep_motifs_a", "rep_motifs_b", "alt_motifs_a", "alt_motifs_b"),
    ),
    "finite_experience_occurrence": (
        ("clean_sequence", "occurrence_probability"),
        ("n_a", "n_b", "occ_n20_a", "occ_n20_b"),
    ),
    "local_representativeness": (
        ("clean_sequence", "periodicity_score", "multiscale_local_imbalance"),
        (
            "p_alts_a", "p_alts_b", "periodicity_a", "periodicity_b",
            "multiscale_imbalance_a", "multiscale_imbalance_b",
        ),
    ),
}
PAIRS = [
    (a, b)
    for n in (1, 2, 3, 4, 5)
    for a, b in product(
        ["".join(s) for s in product("HT", repeat=n)],
        ["".join(s) for s in product("HT", repeat=n)],
    )
]


def _bound_columns(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Data"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def _module_functions(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    return {
        node.name: ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _canonical_sources():
    src = (REPO_ROOT / "src" / "subjective_randomness" / "features.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    return {
        node.name: ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _loaded_hook(model):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_rawfeat_{model}", REGISTRY / f"{model}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_features


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_vendored_helpers_are_byte_identical_to_the_originals(model):
    helpers, _ = VENDORED[model]
    vendored = _module_functions(REGISTRY / f"{model}.py")
    original = _canonical_sources()
    for helper in helpers:
        assert helper in vendored, f"{model} does not vendor {helper}"
        assert vendored[helper] == original[helper], (
            f"{model}'s copy of {helper} has drifted from features.py"
        )


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_compute_features_returns_correct_columns_and_values(model):
    _, columns = VENDORED[model]
    hook = _loaded_hook(model)
    for seq_a, seq_b in PAIRS:
        produced = hook(seq_a, seq_b)
        assert set(produced) == set(columns), (
            f"{model}.compute_features returned {sorted(produced)}, expected {sorted(columns)}"
        )


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_compute_features_covers_every_column_the_model_binds(model):
    _, columns = VENDORED[model]
    bound = _bound_columns(REGISTRY / f"{model}.py")
    assert bound - {"chose_left"} <= set(columns), (
        f"{model} binds {sorted(bound - {'chose_left'} - set(columns))} which "
        "compute_features does not supply"
    )


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_seeds_are_self_contained_and_mirrored(model):
    source = (REGISTRY / f"{model}.py").read_text(encoding="utf-8")
    assert "subjective_randomness.features" not in source
    assert "featurize_stimulus" not in source
    assert (POOL / f"{model}.py").read_text(encoding="utf-8") == source, (
        f"the live pool copy of {model} has drifted from the registry"
    )


def test_motif_stack_needs_no_hook():
    raw = (REGISTRY / "motif_stack.py").read_text(encoding="utf-8")
    assert "def prepare_observed" in raw and "def compute_features" not in raw
    assert raw == (POOL / "motif_stack.py").read_text(encoding="utf-8")


def test_seeding_can_be_pointed_at_the_pool(tmp_path):
    from src.pipelines.outer_loop.orchestrator import seed_experiment_models_from_project

    pool = tmp_path / "seed_models"
    pool.mkdir()
    (pool / "models_manifest.yaml").write_text(
        "models:\n  - name: motif_stack\n    rationale: pool marker\n",
        encoding="utf-8",
    )
    (pool / "motif_stack.py").write_text("# the pool copy\n", encoding="utf-8")

    exp_dir = tmp_path / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)
    assert seed_experiment_models_from_project(
        exp_dir, "subjective_randomness", seed_dir=pool
    )
    seeded = (exp_dir / "cognitive_models" / "motif_stack.py").read_text(encoding="utf-8")
    assert seeded == "# the pool copy\n", "seeded from the default pool, not seed_dir"


def test_seed_exclusion_reads_the_same_pool_that_seeding_uses(tmp_path):
    from src.subjective_randomness.holdout_data import seed_exclusion

    scrubbed = tmp_path / "seed_models"
    scrubbed.mkdir()
    (scrubbed / "models_manifest.yaml").write_text(
        "models:\n  - name: motif_stack\n    rationale: r\n", encoding="utf-8"
    )
    intact = tmp_path / "seed_models_full"
    intact.mkdir()
    (intact / "models_manifest.yaml").write_text(
        "models:\n  - name: motif_stack\n    rationale: r\n"
        "  - name: falk_konold_dp\n    rationale: r\n", encoding="utf-8"
    )

    assert seed_exclusion("falk_konold_dp", scrubbed) == ()
    assert seed_exclusion("falk_konold_dp", intact) == ("falk_konold_dp",)
    assert seed_exclusion("motif_stack", scrubbed) == ("motif_stack",)
