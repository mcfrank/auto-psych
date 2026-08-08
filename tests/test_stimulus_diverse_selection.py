"""Diverse, joint-information stimulus selection.

The naive top-k by per-stimulus EIG can double up on the same model distinction.
``select_informative_stimuli`` instead greedily maximizes the *joint* information
the chosen set carries about model identity, so it spreads across distinctions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.subjective_randomness.stimulus_design import (
    _mean_posterior_entropy,
    _posterior_entropy,
    build_exhaustive_design,
    default_model_family_names,
    enumerate_all_pairs,
    family_predict_fns,
    posterior_param_sets,
    select_discriminating_stimuli,
    select_informative_stimuli,
)

# Three models A/B/C; each stimulus "kind" induces a fixed p_left per model.
#   C_vs_AB / C_vs_AB_2 : near-duplicates, both separate C from {A, B} (high EIG)
#   A_vs_B              : separates A from B (lower marginal EIG, complementary)
_TABLE = {
    "C_vs_AB": {"A": 0.95, "B": 0.95, "C": 0.05},
    "C_vs_AB_2": {"A": 0.96, "B": 0.96, "C": 0.04},
    "A_vs_B": {"A": 0.95, "B": 0.05, "C": 0.50},
}


def _toy_predict_fns():
    def make(model):
        return lambda stim: _TABLE[stim["kind"]][model]

    return {m: make(m) for m in ("A", "B", "C")}


def _stim(kind):
    return {"kind": kind, "sequence_a": "HH", "sequence_b": "TT"}


def test_diverse_selection_picks_complementary_over_redundant():
    predict_fns = _toy_predict_fns()
    stimuli = [_stim("C_vs_AB"), _stim("C_vs_AB_2"), _stim("A_vs_B")]

    selected = select_informative_stimuli(
        stimuli, predict_fns, k=2, n_scenarios=4000, seed=0
    )
    kinds = {s["kind"] for s in selected}
    assert len(selected) == 2
    # The complementary distinction (A vs B) must be covered, not two redundant
    # C-vs-AB probes.
    assert "A_vs_B" in kinds

    # Contrast: naive top-k by marginal EIG doubles up on the redundant pair.
    naive_kinds = {s["kind"] for s in select_discriminating_stimuli(stimuli, predict_fns, k=2)}
    assert naive_kinds == {"C_vs_AB", "C_vs_AB_2"}
    assert "A_vs_B" not in naive_kinds


def test_build_exhaustive_design_enumerates_and_selects_k():
    # Uses the real reference families over a small length range so it stays fast.
    sel = build_exhaustive_design(k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0)
    assert len(sel) == 5
    assert all("sequence_a" in s and "sequence_b" in s for s in sel)
    # All distinct pairs, each within the requested lengths.
    keys = {(s["sequence_a"], s["sequence_b"]) for s in sel}
    assert len(keys) == 5
    assert all(len(s["sequence_a"]) in (3, 4) and len(s["sequence_b"]) in (3, 4) for s in sel)
    # Carries the "eig" field the stimuli.json contract / design validator expects.
    assert all("eig" in s for s in sel)


def _seed_experiment_models(exp_dir):
    """Copy the fixture PyMC models into exp_dir/cognitive_models."""
    import shutil
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "fixtures" / "pymc_models"
    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin.py", "representativeness.py", "models_manifest.yaml"):
        shutil.copyfile(fixture_dir / name, models_dir / name)
    return models_dir


@pytest.mark.slow
def test_write_exhaustive_design_writes_stimuli_json(tmp_path):
    """Experiment 1: exhaustive design scores the experiment's ACTUAL PyMC
    model set from its prior predictive (no pure-Python twins involved)."""
    import json

    from src.pipelines.outer_loop.run import _write_exhaustive_design

    exp_dir = tmp_path / "experiment1"
    _seed_experiment_models(exp_dir)
    _write_exhaustive_design(exp_dir, "subjective_randomness", k=4, lengths=(3, 4))
    data = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 4
    assert all("sequence_a" in s and "sequence_b" in s for s in data)
    assert all("eig" in s and "selection_rank" in s for s in data)


def test_write_exhaustive_design_posterior_wiring(tmp_path, monkeypatch):
    """Experiment >= 2: the previous experiment's responses and registry are
    threaded into design_exhaustive's posterior mode, with the fit cache under
    this experiment's design dir."""
    import json

    from src.pipelines.outer_loop.run import _write_exhaustive_design

    calls: dict = {}

    def fake_design(models_dir, registry_path=None, **kwargs):
        calls["models_dir"] = models_dir
        calls["registry_path"] = registry_path
        calls.update(kwargs)
        return [
            {"sequence_a": "HHH", "sequence_b": "HTH", "eig": 0.5,
             "selection_rank": 1, "joint_eig_bits": 0.5}
        ]

    monkeypatch.setattr(
        "src.pipelines.outer_loop.eig.design_exhaustive", fake_design
    )

    prev = tmp_path / "experiment1"
    (prev / "data").mkdir(parents=True)
    (prev / "data" / "responses.csv").write_text("participant_id\n", encoding="utf-8")
    exp_dir = tmp_path / "experiment2"
    exp_dir.mkdir()

    _write_exhaustive_design(
        exp_dir, "subjective_randomness", exp_num=2, prev_exp_dir=prev,
        k=1, lengths=(3, 4),
    )

    assert calls["models_dir"] == exp_dir / "cognitive_models"
    assert calls["registry_path"] == prev / "model_registry.yaml"
    assert calls["responses_csv"] == prev / "data" / "responses.csv"
    assert calls["fit_cache_dir"] == exp_dir / "design" / "_fit_cache"
    assert calls["n_select"] == 1
    data = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert data[0]["eig"] == 0.5


def test_posterior_param_sets_extracts_named_params_from_idata():
    import numpy as np
    from types import SimpleNamespace

    post = {
        "delta": np.array([[0.10, 0.20], [0.30, 0.40]]),  # (chain, draw)
        "alpha": np.array([[0.50, 0.60], [0.70, 0.80]]),
    }

    class _Post:
        def __contains__(self, k):
            return k in post

        def __getitem__(self, k):
            return SimpleNamespace(values=post[k])

        @property
        def data_vars(self):
            return list(post)

    idata = SimpleNamespace(posterior=_Post())
    sets = posterior_param_sets(idata, ["delta", "alpha"], n_draws=3, seed=0)
    assert len(sets) == 3
    assert all(set(s) == {"delta", "alpha"} for s in sets)
    assert all(round(s["delta"], 2) in {0.10, 0.20, 0.30, 0.40} for s in sets)


def test_posterior_param_sets_fails_loudly_on_missing_variable():
    import numpy as np
    from types import SimpleNamespace

    post = {"delta": np.array([[0.1, 0.2]])}

    class _Post:
        def __contains__(self, k):
            return k in post

        def __getitem__(self, k):
            return SimpleNamespace(values=post[k])

        @property
        def data_vars(self):
            return list(post)

    idata = SimpleNamespace(posterior=_Post())
    try:
        posterior_param_sets(idata, ["delta", "alpha"], n_draws=2)
    except KeyError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected KeyError for missing posterior variable")


def test_build_exhaustive_design_with_explicit_posterior_params():
    import importlib

    names = default_model_family_names()
    param_sets_by_model = {}
    for n in names:
        mod = importlib.import_module(f"src.subjective_randomness.model_families.{n}")
        mid = {p: (lo + hi) / 2 for p, (lo, hi) in mod.PARAM_BOUNDS.items()}
        param_sets_by_model[n] = [mid, dict(mid)]

    sel = build_exhaustive_design(
        k=4,
        lengths=(3, 4),
        param_sets_by_model=param_sets_by_model,
        n_scenarios=200,
        prefilter=150,
        seed=0,
    )
    assert len(sel) == 4
    assert all("sequence_a" in s and "sequence_b" in s for s in sel)


def test_legacy_compat_reproduces_the_historical_algorithm():
    """The single most important test in this module: legacy_compat=True must
    still run the original algorithm (no quotienting, the original shared-seed
    prior-draw RNG, plain full-scan greedy), proving every fast-path change since
    (chunked entropy, CELF, the sequence-class quotient, the streamed pair scan,
    score decomposition) is additive -- the old path is untouched, not just
    "close" to its former output.

    This pins the algorithm's machine-stable signature rather than a byte-exact
    pair list. The greedy loop ranks candidates by a Monte Carlo EIG estimate
    whose exact bits depend on libm's log/exp rounding, and the lengths-(3, 4)
    universe is full of complement-symmetric pairs whose true EIGs are exactly
    equal -- so which member of such a tie argsort returns can differ between
    machines on a last-ulp difference, with no numerical error anywhere. The
    original byte-exact pin was generated on a different machine and has never
    reproduced on this one (verified failing at 1c8436c, the commit that
    introduced it). What IS stable, and is asserted below: the multiset of EIG
    values (tie partners share an EIG by construction, so swapping one changes
    nothing), the selections that are tie-free by a wide margin, the selection
    bookkeeping, the candidate universe, and determinism.
    """
    sel = build_exhaustive_design(
        k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0, legacy_compat=True
    )

    assert len(sel) == 5
    assert [s["selection_order"] for s in sel] == [0, 1, 2, 3, 4]

    pairs = [(s["sequence_a"], s["sequence_b"]) for s in sel]
    assert len(set(pairs)) == 5
    for a, b in pairs:
        assert len(a) in (3, 4) and len(b) in (3, 4)
        assert set(a) <= {"H", "T"} and set(b) <= {"H", "T"}
    assert all(s["eig"] > 0 for s in sel)

    # The historical pin's own EIG multiset, which survives the tie-breaking
    # differences that made its pair list unportable.
    assert sorted(s["eig"] for s in sel) == pytest.approx(
        [0.120592, 0.125002, 0.125002, 0.149885, 0.149885], abs=1e-6
    )

    # The first selection is not a tie: its mean posterior entropy beats the
    # runner-up's by a relative 4.6e-3 (measured), a margin no libm rounding
    # difference can cross, and exactly one candidate attains the minimum.
    assert pairs[0] == ("HTH", "HTTH")

    repeat = build_exhaustive_design(
        k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0, legacy_compat=True
    )
    assert sel == repeat


def test_mean_posterior_entropy_is_bit_identical_to_unchunked_formula():
    rng = np.random.default_rng(0)
    n_scenarios, n_pool, n_models = 64, 300, 3
    log_belief = np.log(rng.dirichlet(np.ones(n_models), size=n_scenarios))
    logP = np.log(rng.uniform(1e-6, 1 - 1e-6, size=(n_pool, n_models)))
    log1mP = np.log(1.0 - np.exp(logP))
    responses = (rng.random((n_scenarios, n_pool)) < rng.random((n_scenarios, n_pool))).astype(float)
    idx = np.arange(n_pool)

    def unchunked(log_belief, responses, logP, log1mP, idx):
        r = responses[:, idx]
        contrib = r[:, :, None] * logP[idx][None] + (1.0 - r)[:, :, None] * log1mP[idx][None]
        tentative = log_belief[:, None, :] + contrib
        return _posterior_entropy(tentative).mean(axis=0)

    ref = unchunked(log_belief, responses, logP, log1mP, idx)
    for chunk in (1, 7, 300, 10_000):
        got = _mean_posterior_entropy(log_belief, responses, logP, log1mP, idx, chunk=chunk)
        assert np.array_equal(ref, got), f"chunk={chunk} not bit-identical"

    # Also with a non-contiguous subset, as the greedy loop's `remaining` array is.
    subset = np.array(sorted(rng.choice(n_pool, size=100, replace=False)))
    ref_sub = unchunked(log_belief, responses, logP, log1mP, subset)
    got_sub = _mean_posterior_entropy(log_belief, responses, logP, log1mP, subset, chunk=17)
    assert np.array_equal(ref_sub, got_sub)


# --- CELF (lazy=True) --------------------------------------------------------


def test_lazy_defaults_to_off_and_matches_full_scan_by_construction():
    """lazy=False (the default) must run the exact same code path as before this
    change -- select_informative_stimuli's default output must be untouched."""
    predict_fns = _toy_predict_fns()
    stimuli = [_stim("C_vs_AB"), _stim("C_vs_AB_2"), _stim("A_vs_B")]
    default = select_informative_stimuli(stimuli, predict_fns, k=2, n_scenarios=2000, seed=0)
    explicit_off = select_informative_stimuli(
        stimuli, predict_fns, k=2, n_scenarios=2000, seed=0, lazy=False
    )
    assert [s["kind"] for s in default] == [s["kind"] for s in explicit_off]


def test_default_path_deliberately_differs_from_legacy_compat():
    """The default (fast) build_exhaustive_design path quotients the sequence
    space and uses CELF by default -- it is expected, not a regression, that its
    output differs from legacy_compat=True's. This test exists so that fact is
    asserted deliberately rather than discovered by a future golden-test
    failure someone has to puzzle out."""
    default = build_exhaustive_design(k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0)
    legacy = build_exhaustive_design(
        k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0, legacy_compat=True
    )
    default_keys = [(s["sequence_a"], s["sequence_b"]) for s in default]
    legacy_keys = [(s["sequence_a"], s["sequence_b"]) for s in legacy]
    assert default_keys != legacy_keys
    # Still a valid design either way: right size, distinct pairs, positive EIG.
    assert len(default) == 5
    assert len(set(default_keys)) == 5
    assert all(s["eig"] > 0 for s in default)


def test_default_path_scales_past_the_old_length_12_cap():
    """The legacy path materializes every pair up front -- length 12 needed
    ~15GB and was never actually run. The default path streams the scan, so
    this must complete (previously impossible, not just slow)."""
    sel = build_exhaustive_design(k=8, lengths=range(2, 13), seed=0)
    assert len(sel) == 8
    assert len({(s["sequence_a"], s["sequence_b"]) for s in sel}) == 8
    assert all(s["eig"] > 0 for s in sel)
    assert all(2 <= len(s["sequence_a"]) <= 12 and 2 <= len(s["sequence_b"]) <= 12 for s in sel)


def test_max_length_is_threaded_to_sequence_classes():
    """max_length is build_sequence_classes' own guard against runaway
    enumeration -- confirm build_exhaustive_design actually forwards it rather
    than silently using its own default."""
    with pytest.raises(ValueError, match="max_length"):
        build_exhaustive_design(k=4, lengths=(10,), max_length=8)


def _register_fake_family(monkeypatch, *, model_name, sufficient_stats, complement_invariant, score_fn):
    """Install a fake model_families module in sys.modules under `model_name` so
    build_exhaustive_design's importlib.import_module(...) call picks it up,
    exactly as it would a real family."""
    import types

    from src.subjective_randomness.model_families.common import (
        choice_probability,
        merge_params,
        normalize_stimulus,
    )

    module = types.ModuleType(f"src.subjective_randomness.model_families.{model_name}")
    module.MODEL_NAME = model_name
    module.DEFAULT_PARAMS = {"beta": 4.0, "side_bias": 0.0}
    module.PARAM_BOUNDS = {"beta": (0.2, 12.0), "side_bias": (-2.0, 2.0)}
    module.SUFFICIENT_STATS = sufficient_stats
    module.COMPLEMENT_INVARIANT = complement_invariant
    module.score_sequence = score_fn

    def predict_left(stimulus, params=None):
        seq_a, seq_b = normalize_stimulus(stimulus)
        p = merge_params(module.DEFAULT_PARAMS, params)
        return choice_probability(score_fn(seq_a, p), score_fn(seq_b, p), p)

    module.predict_left = predict_left
    monkeypatch.setitem(
        __import__("sys").modules,
        f"src.subjective_randomness.model_families.{model_name}",
        module,
    )


def test_on_quotient_violation_fallback_still_returns_a_valid_design(monkeypatch):
    """A model that lies about SUFFICIENT_STATS (declares n/max_run, actually
    reads seq[0]) must not crash the default path -- it should warn and fall
    back to the identity quotient (no merging), which is always safe."""
    _register_fake_family(
        monkeypatch,
        model_name="_test_lying_position_reader",
        sufficient_stats=("n", "max_run"),
        complement_invariant=False,
        score_fn=lambda seq, params=None: 1.0 if seq[0] == "H" else 0.0,
    )
    sel = build_exhaustive_design(
        k=4,
        lengths=(3, 4),
        model_names=["_test_lying_position_reader"],
        n_scenarios=100,
        prefilter=50,
        seed=0,
        on_quotient_violation="fallback",
    )
    assert len(sel) == 4
    assert len({(s["sequence_a"], s["sequence_b"]) for s in sel}) == 4


def test_on_quotient_violation_raise_propagates(monkeypatch):
    from src.subjective_randomness.exhaustive_search import QuotientViolation

    _register_fake_family(
        monkeypatch,
        model_name="_test_lying_position_reader_2",
        sufficient_stats=("n", "max_run"),
        complement_invariant=False,
        score_fn=lambda seq, params=None: 1.0 if seq[0] == "H" else 0.0,
    )
    with pytest.raises(QuotientViolation):
        build_exhaustive_design(
            k=4,
            lengths=(3, 4),
            model_names=["_test_lying_position_reader_2"],
            n_scenarios=100,
            prefilter=50,
            seed=0,
            on_quotient_violation="raise",
        )


def test_on_quotient_violation_rejects_unknown_policy(monkeypatch):
    _register_fake_family(
        monkeypatch,
        model_name="_test_lying_position_reader_3",
        sufficient_stats=("n", "max_run"),
        complement_invariant=False,
        score_fn=lambda seq, params=None: 1.0 if seq[0] == "H" else 0.0,
    )
    with pytest.raises(ValueError, match="Unknown on_quotient_violation"):
        build_exhaustive_design(
            k=4,
            lengths=(3, 4),
            model_names=["_test_lying_position_reader_3"],
            n_scenarios=100,
            prefilter=50,
            seed=0,
            on_quotient_violation="nonsense",
        )


def test_a_well_declared_model_never_triggers_the_fallback(monkeypatch, capsys):
    """Sanity check the negative of the two tests above: a model whose
    declaration is actually correct must not print the fallback warning.

    A model declaring only "n" collapses every same-length sequence into one
    class (it genuinely cannot distinguish them) -- with lengths=(3,4) that
    leaves exactly one possible pair, so k=1 here, not a larger number (see
    test_raises_a_clear_error_when_the_quotiented_pool_is_smaller_than_k for
    what happens if you ask for more than the quotient has to offer)."""
    _register_fake_family(
        monkeypatch,
        model_name="_test_honest_length_reader",
        sufficient_stats=("n",),
        complement_invariant=True,
        score_fn=lambda seq, params=None: float(len(seq)),
    )
    sel = build_exhaustive_design(
        k=1,
        lengths=(3, 4),
        model_names=["_test_honest_length_reader"],
        n_scenarios=100,
        prefilter=50,
        seed=0,
    )
    assert len(sel) == 1
    assert "quotient audit failed" not in capsys.readouterr().out


def test_raises_a_clear_error_when_the_quotiented_pool_is_smaller_than_k(monkeypatch):
    """The bug this guards against: a narrow quotient can legitimately collapse
    the space to fewer distinct pairs than k, which must fail with a clear
    message (matching select_informative_stimuli's own convention of raising
    rather than silently returning fewer than requested), not crash inside the
    greedy loop once it runs out of candidates."""
    _register_fake_family(
        monkeypatch,
        model_name="_test_honest_length_reader_2",
        sufficient_stats=("n",),
        complement_invariant=True,
        score_fn=lambda seq, params=None: float(len(seq)),
    )
    with pytest.raises(ValueError, match="Only 1 distinct class-pair"):
        build_exhaustive_design(
            k=4,
            lengths=(3, 4),
            model_names=["_test_honest_length_reader_2"],
            n_scenarios=100,
            prefilter=50,
            seed=0,
        )


def test_celf_matches_full_scan_on_a_clean_toy_case():
    """On the well-separated toy table (large gaps between candidates' gains,
    plenty of scenarios), CELF must select identically to the full scan."""
    predict_fns = _toy_predict_fns()
    stimuli = [_stim("C_vs_AB"), _stim("C_vs_AB_2"), _stim("A_vs_B")]
    plain = select_informative_stimuli(stimuli, predict_fns, k=2, n_scenarios=4000, seed=0)
    lazy = select_informative_stimuli(
        stimuli, predict_fns, k=2, n_scenarios=4000, seed=0, lazy=True, lazy_audit=True
    )
    assert [s["kind"] for s in plain] == [s["kind"] for s in lazy]


def test_celf_is_deterministic_given_seed():
    names = default_model_family_names()
    fns = family_predict_fns(names, param_samples=30, seed=2)
    cands = enumerate_all_pairs((3, 4, 5))
    a = select_informative_stimuli(cands, fns, k=6, n_scenarios=500, prefilter=200, seed=2, lazy=True)
    b = select_informative_stimuli(cands, fns, k=6, n_scenarios=500, prefilter=200, seed=2, lazy=True)
    assert [(s["sequence_a"], s["sequence_b"]) for s in a] == [
        (s["sequence_a"], s["sequence_b"]) for s in b
    ]


def test_celf_returns_a_valid_k_sized_distinct_selection_even_when_it_diverges():
    """Documented finding: CELF can disagree with the full-scan greedy under this
    finite-scenario estimator (measured up to ~1/3 of trials at small
    n_scenarios). Whether or not it disagrees on a given seed, the output must
    still be a valid size-k selection of distinct candidates."""
    names = default_model_family_names()
    fns = family_predict_fns(names, param_samples=30, seed=3)
    cands = enumerate_all_pairs((3, 4, 5))
    out = select_informative_stimuli(
        cands, fns, k=8, n_scenarios=300, prefilter=300, seed=3, lazy=True
    )
    assert len(out) == 8
    keys = {(s["sequence_a"], s["sequence_b"]) for s in out}
    assert len(keys) == 8
    assert all("eig" in s and "selection_order" in s for s in out)


def test_lazy_audit_detects_a_known_divergence():
    """Pins a concrete (seed, params) case (found empirically) where CELF
    disagrees with the full-scan greedy under n_scenarios=300, proving
    lazy_audit actually catches real divergence rather than always passing."""
    names = default_model_family_names()
    fns = family_predict_fns(names, param_samples=30, seed=1)
    cands = enumerate_all_pairs((3, 4, 5))
    try:
        select_informative_stimuli(
            cands, fns, k=8, n_scenarios=300, prefilter=300, seed=1, lazy=True, lazy_audit=True
        )
    except AssertionError as exc:
        assert "CELF disagreed with the full-scan greedy" in str(exc)
    else:
        raise AssertionError(
            "expected lazy_audit=True to catch a known CELF/full-scan divergence "
            "at this (seed, n_scenarios) -- if this no longer reproduces, the "
            "audit mechanism itself may have broken silently."
        )


def test_selection_is_deterministic_and_returns_k_from_input():
    predict_fns = _toy_predict_fns()
    stimuli = [_stim("C_vs_AB"), _stim("C_vs_AB_2"), _stim("A_vs_B")]
    valid_kinds = {s["kind"] for s in stimuli}

    a = select_informative_stimuli(stimuli, predict_fns, k=2, n_scenarios=2000, seed=1)
    b = select_informative_stimuli(stimuli, predict_fns, k=2, n_scenarios=2000, seed=1)
    assert [s["kind"] for s in a] == [s["kind"] for s in b]
    assert len(a) == 2
    assert all(s["kind"] in valid_kinds for s in a)
