"""In a raw-features run the agents' CSV carries only H/T sequences, so each
seed must compute the columns it binds itself (see docs/raw_features_arm.md).

These are the guards that make that safe:

* the helpers vendored into each seed are byte-identical to the originals in
  `src/subjective_randomness/features.py`, and behave identically;
* each seed's `compute_features` returns exactly the columns its `pm.Data`
  containers read, with the values the featurizer would have supplied. The
  model body is unchanged and reads those columns by name, so equal columns
  mean identical model input and therefore identical predictions;
* no seed imports the project featurizer (which a raw-features run puts out of
  reach), and the registry and live pool copies stay in step.
"""

from __future__ import annotations

import ast
from itertools import product

import pytest

from src.subjective_randomness import features as canonical
from tests.paths import REPO_ROOT

# The raw-features seed set is SEPARATE from the featurized one: a model that
# computes a column the CSV already carries makes the fit raise (and the loop
# only [drop]s it), so the two cannot share files. `raw_features` runs point
# `seed_models_dir` / the live pool at these `_raw` directories.
REGISTRY = REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families_raw"
POOL = (
    REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects"
    / "subjective_randomness" / "seed_models_raw"
)
FEATURIZED_REGISTRY = (
    REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families"
)
# model -> (vendored helpers, the columns its compute_features must return)
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
# Every distinct pair over these lengths: small, exhaustive, and covers the
# length-1 and all-same-symbol edge cases the featurizer guards.
PAIRS = [
    (a, b)
    for n in (1, 2, 3, 4, 5)
    for a, b in product(
        ["".join(s) for s in product("HT", repeat=n)],
        ["".join(s) for s in product("HT", repeat=n)],
    )
]


def _bound_columns(path):
    """Every column name the model binds with `pm.Data("name", ...)`.

    Parsed, not grepped: `local_representativeness` splits the call across
    lines, so a line-oriented pattern silently misses two of its columns —
    which is how the first draft of this arm shipped an incomplete
    `compute_features`.
    """
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
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.get_source_segment(path.read_text(encoding="utf-8"), node)
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
def test_compute_features_matches_the_featurizer_on_the_columns_it_returns(model):
    _, columns = VENDORED[model]
    hook = _loaded_hook(model)
    for seq_a, seq_b in PAIRS:
        produced = hook(seq_a, seq_b)
        assert set(produced) == set(columns), (
            f"{model}.compute_features returned {sorted(produced)}, expected {sorted(columns)}"
        )
        expected = canonical.featurize_stimulus(seq_a, seq_b)
        for column in columns:
            assert produced[column] == pytest.approx(expected[column]), (
                f"{model}.compute_features[{column}] disagrees with the featurizer "
                f"on ({seq_a}, {seq_b})"
            )


def _loaded_hook(model):
    """The model's compute_features, loaded without building its PyMC graph."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_rawfeat_{model}", REGISTRY / f"{model}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_features


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_compute_features_covers_every_column_the_model_binds(model):
    """A column the model binds but does not compute would be missing from a
    raw CSV, so the fit would die inside the sweep rather than here."""
    _, columns = VENDORED[model]
    bound = _bound_columns(REGISTRY / f"{model}.py")
    # chose_left is the observed response, always present in a raw CSV.
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


@pytest.mark.parametrize("model", sorted(VENDORED))
def test_raw_seed_is_its_featurized_twin_plus_the_appended_block(model):
    """The two seed sets must not drift apart: a raw seed is exactly its
    featurized counterpart with the self-contained block appended. Anything
    else means the arm is testing a different model, not the same model on
    raw data."""
    featurized = (FEATURIZED_REGISTRY / f"{model}.py").read_text(encoding="utf-8")
    raw = (REGISTRY / f"{model}.py").read_text(encoding="utf-8")
    assert raw.startswith(featurized.rstrip("\n") + "\n"), (
        f"{model}: the raw seed no longer starts with its featurized twin"
    )
    assert "def compute_features" not in featurized
    assert "Raw-features (arm C) support" in raw[len(featurized.rstrip("\n")):]


def test_the_featurized_seed_set_is_untouched_by_this_arm():
    """A raw-features run must not change the featurized path: those seeds are
    fitted on CSVs that already carry the columns, where `compute_features`
    would collide and the loop would silently drop the model."""
    for path in sorted(FEATURIZED_REGISTRY.glob("*.py")):
        if path.name == "__init__.py":
            continue
        assert "def compute_features" not in path.read_text(encoding="utf-8"), (
            f"{path.name} on the featurized path declares compute_features"
        )


def test_motif_stack_needs_no_hook_in_either_set():
    """It builds every array from the raw sequences in `prepare_observed`
    already, which is why it is copied verbatim."""
    raw = (REGISTRY / "motif_stack.py").read_text(encoding="utf-8")
    assert "def prepare_observed" in raw and "def compute_features" not in raw
    assert raw == (FEATURIZED_REGISTRY / "motif_stack.py").read_text(encoding="utf-8")
