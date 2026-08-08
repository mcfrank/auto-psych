"""Fast unit tests for the subjective_randomness featurizer (no PyMC)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_sr_preprocess", PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_featurize_stimulus_counts_heads_alternations_runs():
    pp = _load()
    feats = pp.featurize_stimulus("HHHT", "HTHT")

    assert feats["n_a"] == 4 and feats["n_b"] == 4
    assert feats["h_a"] == 3 and feats["h_b"] == 2
    assert feats["alts_a"] == 1  # HHHT: one H→T transition
    assert feats["alts_b"] == 3  # HTHT: H-T-H-T, three transitions
    assert feats["max_run_a"] == 3  # "HHH"
    assert feats["max_run_b"] == 1
    assert feats["p_a"] == pytest.approx(0.75)
    assert feats["p_alts_b"] == pytest.approx(1.0)
    assert feats["imbalance_a"] == pytest.approx(0.5)
    assert feats["imbalance_b"] == pytest.approx(0.0)
    assert feats["max_run_norm_a"] == pytest.approx(2.0 / 3.0)
    assert feats["max_run_norm_b"] == pytest.approx(0.0)
    assert feats["periodicity_a"] == pytest.approx(0.5)
    assert feats["periodicity_b"] == pytest.approx(1.0)


def test_parse_motifs_matches_falk_konold_examples():
    # Motif parsing lives canonically in features.py now (preprocess.py re-exports
    # only featurize_stimulus), so test it at its source.
    from src.subjective_randomness.features import parse_motifs

    # Falk & Konold (1997) Difficulty Predictor parse: HHTTHTHT -> two runs
    # (HH, TT) plus one alternating sub-sequence (HTHT), so DP = 2*1 + 1*2 = 4.
    assert parse_motifs("HHTTHTHT") == (2, 1)
    # A fully alternating sequence is a single alternation motif.
    assert parse_motifs("HTHTHT") == (0, 1)
    # A single constant run is one repetition motif.
    assert parse_motifs("HHHHHH") == (1, 0)
    # An isolated single symbol between runs is itself a repetition motif.
    assert parse_motifs("HHTHH") == (3, 0)
    # A length-2 alternation (one transition) counts as one alternation motif.
    assert parse_motifs("HHTH") == (1, 1)
    # Degenerate inputs.
    assert parse_motifs("H") == (1, 0)
    assert parse_motifs("") == (0, 0)


def test_parse_motifs_uses_minimal_falk_konold_parse():
    # Falk & Konold (1997, p. 308) define the Difficulty Predictor over the
    # partition that "achieve[s] the lowest possible number", and their printed
    # example splits a pure run to extend an alternating chunk (XXXOXO ->
    # XX|XOXO). A greedy maximal-runs parse gets these wrong.
    from src.subjective_randomness.features import parse_motifs

    # (HTH)(HTH): DP 4, not the greedy 5 from rep(HH-split) + alt chunks.
    assert parse_motifs("HTHHTH") == (0, 2)
    # (HH)(THT)(THT): DP 5, not the greedy 6.
    assert parse_motifs("HHTHTTHT") == (1, 2)
    # DP ties resolve to the fewest chunks (most compressed description).
    assert parse_motifs("HTHT") == (0, 1)


def test_parse_motifs_implementations_agree_exhaustively():
    # features.py and model_families/common.py deliberately mirror the parse;
    # they must agree on every H/T sequence up to length 8.
    from itertools import product

    from src.subjective_randomness.features import parse_motifs as parse_feat
    from src.subjective_randomness.model_families.common import (
        parse_motifs as parse_family,
    )

    for n in range(1, 9):
        for bits in product("HT", repeat=n):
            seq = "".join(bits)
            assert parse_feat(seq) == parse_family(seq), seq


def test_featurize_stimulus_adds_motif_parse_counts():
    pp = _load()
    feats = pp.featurize_stimulus("HHTTHTHT", "HTHTHTHT")
    assert feats["rep_motifs_a"] == 2 and feats["alt_motifs_a"] == 1
    assert feats["rep_motifs_b"] == 0 and feats["alt_motifs_b"] == 1


def test_featurize_stimulus_adds_per_symbol_columns():
    # The motif-HMM model (Griffiths et al. 2018, marginalised forward pass)
    # needs the raw symbols: sym1..sym8 as 0/1 (H=1), zero-padded past n.
    pp = _load()
    feats = pp.featurize_stimulus("HHT", "THTH")
    assert [feats[f"sym{i}_a"] for i in range(1, 9)] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert [feats[f"sym{i}_b"] for i in range(1, 9)] == [0, 1, 0, 1, 0, 0, 0, 0]


def test_featurize_stimulus_adds_occurrence_columns():
    # Hahn & Warren (2009) occurrence probabilities within finite global
    # windows of 10/20/50 flips. HHHH values follow from their footnote 1
    # (nonoccurrence 0.75 / 0.52 / 0.17 to 2 d.p.).
    pp = _load()
    feats = pp.featurize_stimulus("HHHH", "HT")
    assert feats["occ_n10_a"] == pytest.approx(0.25, abs=0.005)
    assert feats["occ_n20_a"] == pytest.approx(0.48, abs=0.005)
    assert feats["occ_n50_a"] == pytest.approx(0.83, abs=0.005)
    # HT is all but certain to appear somewhere in 20 flips.
    assert feats["occ_n20_b"] > 0.99


def test_occurrence_implementations_agree():
    from itertools import product

    from src.subjective_randomness.features import occurrence_probability as occ_feat
    from src.subjective_randomness.model_families.common import (
        occurrence_probability as occ_family,
    )

    for n in (2, 5, 8):
        for bits in product("HT", repeat=n):
            seq = "".join(bits)
            for window in (10, 20, 50):
                assert occ_feat(seq, window) == pytest.approx(
                    occ_family(seq, window), abs=1e-12
                ), (seq, window)


def test_featurize_stimulus_adds_local_imbalance():
    # Kahneman & Tversky (1972, p. 435): a representative sequence is balanced
    # "not only globally in the entire sample, but also locally in each of its
    # parts". local_imbalance = worst H/T imbalance over sliding windows of
    # length min(n, 4).
    pp = _load()
    feats = pp.featurize_stimulus("HHHHTTTT", "HHTTHHTT")
    # Globally balanced but locally clumped: the HHHH window is all heads.
    assert feats["local_imbalance_a"] == pytest.approx(1.0)
    # Every length-4 window of HHTTHHTT holds two of each.
    assert feats["local_imbalance_b"] == pytest.approx(0.0)
    # Shorter than the window: falls back to whole-sequence imbalance.
    short = pp.featurize_stimulus("HHT", "HT")
    assert short["local_imbalance_a"] == pytest.approx(1.0 / 3.0)
    assert short["local_imbalance_b"] == pytest.approx(0.0)


def test_featurize_stimulus_rejects_sequences_longer_than_max():
    pp = _load()
    with pytest.raises(ValueError, match="longer than"):
        pp.featurize_stimulus("HHHHHHHHH", "HT")  # length 9 > 8


def test_featurize_stimulus_keys_match_pm_data_names():
    pp = _load()
    feats = pp.featurize_stimulus("HT", "TH")
    expected = {
        "n_a",
        "h_a",
        "alts_a",
        "max_run_a",
        "rep_motifs_a",
        "alt_motifs_a",
        "p_a",
        "p_alts_a",
        "max_run_norm_a",
        "imbalance_a",
        "periodicity_a",
        "n_b",
        "h_b",
        "alts_b",
        "max_run_b",
        "rep_motifs_b",
        "alt_motifs_b",
        "p_b",
        "p_alts_b",
        "max_run_norm_b",
        "imbalance_b",
        "periodicity_b",
    }
    expected |= {f"sym{i}_{side}" for i in range(1, 9) for side in ("a", "b")}
    expected |= {f"occ_n{w}_{side}" for w in (10, 20, 50) for side in ("a", "b")}
    expected |= {"local_imbalance_a", "local_imbalance_b"}
    assert set(feats) == expected


def test_featurize_stimulus_handles_length_one():
    pp = _load()
    feats = pp.featurize_stimulus("H", "T")
    assert feats["alts_a"] == 0
    assert feats["p_alts_a"] == 0.0  # n-1 == 0 guard
    assert feats["p_a"] == 1.0
    assert feats["p_b"] == 0.0
    assert feats["max_run_norm_a"] == 0.0
    assert feats["periodicity_a"] == 0.0
