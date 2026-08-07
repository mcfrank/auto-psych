"""Diverse, joint-information stimulus selection.

The naive top-k by per-stimulus EIG can double up on the same model distinction.
``select_informative_stimuli`` instead greedily maximizes the *joint* information
the chosen set carries about model identity, so it spreads across distinctions.
"""

from __future__ import annotations

import numpy as np

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


def test_write_exhaustive_design_writes_stimuli_json(tmp_path):
    import json

    from src.pipelines.outer_loop.run import _write_exhaustive_design

    exp_dir = tmp_path / "experiment1"
    exp_dir.mkdir()
    _write_exhaustive_design(exp_dir, "subjective_randomness", k=4, lengths=(3, 4))
    data = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 4
    assert all("sequence_a" in s and "sequence_b" in s for s in data)


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


def test_build_exhaustive_design_golden_output_unchanged_by_chunked_entropy():
    """Pins the exact output of the greedy loop with today's real families,
    unchanged before and after replacing the unchunked (n_scenarios, R, K) tensor
    computation with the chunked _mean_posterior_entropy helper -- proof this was
    a pure speed/memory refactor, not a behavior change."""
    sel = build_exhaustive_design(k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0)
    assert [
        (s["sequence_a"], s["sequence_b"], s["eig"], s["selection_order"]) for s in sel
    ] == [
        ("HTH", "HTTH", 0.149885, 0),
        ("TTT", "HHHH", 0.125002, 1),
        ("HHH", "TTTT", 0.125002, 2),
        ("THT", "HHTT", 0.120592, 3),
        ("THT", "THHT", 0.149885, 4),
    ]


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


def test_build_exhaustive_design_golden_is_untouched_by_adding_celf():
    """build_exhaustive_design doesn't pass lazy= anywhere -- adding the lazy
    option must not change its (already pinned) default output."""
    sel = build_exhaustive_design(k=5, lengths=(3, 4), n_scenarios=200, prefilter=200, seed=0)
    assert [
        (s["sequence_a"], s["sequence_b"], s["eig"], s["selection_order"]) for s in sel
    ] == [
        ("HTH", "HTTH", 0.149885, 0),
        ("TTT", "HHHH", 0.125002, 1),
        ("HHH", "TTTT", 0.125002, 2),
        ("THT", "HHTT", 0.120592, 3),
        ("THT", "THHT", 0.149885, 4),
    ]


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
