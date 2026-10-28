"""The literature models' computations reproduce values printed in the papers.

Falk & Konold (1997) parses and Difficulty Predictor values; Hahn & Warren
(2009) footnote-1 probabilities and Figure 3A wait times; Kahneman & Tversky's
locality of balance. These are the functions the PyMC models compute their
features with (``src/subjective_randomness/pymc_model_families``). (They were
checked against pure-Python twins of the models until 2026-10-18.)
"""

import itertools

import pytest

from src.subjective_randomness.pymc_model_families.falk_konold_dp import parse_motifs
from src.subjective_randomness.pymc_model_families.finite_experience_occurrence import (
    occurrence_probability,
)
from src.subjective_randomness.pymc_model_families.local_representativeness import (
    multiscale_local_imbalance,
)


def _dp(seq: str) -> int:
    rep, alt = parse_motifs(seq)
    return rep + 2 * alt


class TestFalkKonoldDP:
    def test_the_parser_finds_the_global_minimum(self):
        # HTH | HTH is two alternating chunks (DP = 4). A greedy parser that
        # commits at the central HH instead returns DP = 5.
        assert parse_motifs("HTHHTH") == (0, 2)

    def test_the_difficulty_predictor_matches_the_papers(self):
        # Griffiths et al. (2018)'s worked example of the F&K parse:
        # HHTTHTHT -> (HH)(TT)(HTHT), DP = 4.
        assert _dp("HHTTHTHT") == 4
        # Falk & Konold (1997, pp. 308-309) printed length-21 examples
        # (X -> H, O -> T): P(A)=.20 parses to five pure runs, DP = 5;
        # P(A)=.80 parses to five chunks, two alternating, DP = 7.
        assert _dp("HHHHHHTTTHHTTTTTTTHHH") == 5
        assert _dp("HHTHTHTHTHTTTHHTHTHTH") == 7

    def test_dp_is_not_length_normalised(self):
        # F&K never normalise DP by length: a short and a long sequence with
        # the same parse structure score identically.
        assert _dp("HHTT") == _dp("HHHHTTTT")

    def test_every_short_sequence_parses(self):
        for length in range(1, 9):
            for symbols in itertools.product("HT", repeat=length):
                rep, alt = parse_motifs("".join(symbols))
                assert rep >= 0 and alt >= 0 and rep + alt >= 1


class TestOccurrenceProbability:
    def test_matches_hahn_warren_footnote_1(self):
        # H&W (2009) footnote 1: P(no run of 4 heads in n tosses) =
        # F4_{n+2}/2^n, given to 2 d.p. for n in {5,10,15,20,50,100}.
        expected_nonoccurrence = {5: 0.91, 10: 0.75, 15: 0.63, 20: 0.52, 50: 0.17, 100: 0.03}
        for n, q in expected_nonoccurrence.items():
            assert 1.0 - occurrence_probability("HHHH", n) == pytest.approx(q, abs=0.005)

    def test_wait_times_match_hahn_warren_figure_3a(self):
        # Expected wait time = sum over n >= 0 of P(not yet occurred by n).
        # H&W p. 455-456 / Figure 3A: HHH=14, HHT=8, HHHH=30, HHHT=16, HTHT=20.
        # Truncating at n=400 leaves ~1e-5 of tail mass.
        def wait_time(pattern: str) -> float:
            return sum(1.0 - occurrence_probability(pattern, n) for n in range(400))

        assert wait_time("HHH") == pytest.approx(14.0, abs=1e-3)
        assert wait_time("HHT") == pytest.approx(8.0, abs=1e-3)
        assert wait_time("HHHH") == pytest.approx(30.0, abs=1e-3)
        assert wait_time("HHHT") == pytest.approx(16.0, abs=1e-3)
        assert wait_time("HTHT") == pytest.approx(20.0, abs=1e-3)

    def test_degenerate_windows(self):
        # A pattern cannot occur in a window shorter than itself; a window
        # exactly its length contains it with probability (1/2)^k.
        assert occurrence_probability("HTH", 2) == 0.0
        assert occurrence_probability("HTH", 3) == pytest.approx(0.125)


class TestLocalRepresentativeness:
    def test_balance_is_aggregated_across_local_scales(self):
        # K&T (1972, p. 435): balance must hold "locally in each of its parts".
        assert multiscale_local_imbalance("HHHHTTTT") > multiscale_local_imbalance("HHTTHHTT")
