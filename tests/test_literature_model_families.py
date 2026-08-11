"""Fast unit tests for the literature-faithful pure-Python model families.

Test vectors come straight from the source papers where possible (Falk &
Konold 1997 printed parses; Hahn & Warren 2009 footnote-1 probabilities and
wait times; Griffiths et al. 2018 model equations).
"""

import itertools
import math

import pytest


def _hmm_matrices(delta: float, alpha: float):
    """Independent transcription of Griffiths et al. (2018) Eq. 9 for tests.

    States (0-indexed): 0=H-repeat, 1=T-repeat, 2=HT-motif(H), 3=HT-motif(T),
    4=TH-motif(H), 5=TH-motif(T). Row-normalised per their footnote 11 (C
    dropped, rows divided by their sums). Emissions: even states emit H, odd
    states emit T. Initial vector proportional to (α, α, α², 0, 0, α²).
    """
    a, a2, d = alpha, alpha**2, delta
    rows = [
        [d, a, a2, 0.0, 0.0, a2],
        [a, d, a2, 0.0, 0.0, a2],
        [a, a, 0.0, d, 0.0, a2],
        [a, a, d, 0.0, 0.0, a2],
        [a, a, a2, 0.0, 0.0, d],
        [a, a, a2, 0.0, d, 0.0],
    ]
    transition = [[v / sum(row) for v in row] for row in rows]
    init_raw = [a, a, a2, 0.0, 0.0, a2]
    init = [v / sum(init_raw) for v in init_raw]
    return init, transition


def _brute_force_p_regular(seq: str, delta: float, alpha: float) -> float:
    """Sum P(x, z) over every hidden state path (Eq. 8 by enumeration)."""
    init, transition = _hmm_matrices(delta, alpha)
    emits = "HTHTHT"
    total = 0.0
    n = len(seq)
    for path in itertools.product(range(6), repeat=n):
        if any(emits[s] != seq[t] for t, s in enumerate(path)):
            continue
        p = init[path[0]]
        for t in range(1, n):
            p *= transition[path[t - 1]][path[t]]
        total += p
    return total


class TestFalkKonoldDP:
    def test_pure_python_parser_finds_global_minimum(self):
        from src.subjective_randomness.model_families.common import parse_motifs

        # HTH | HTH is two alternating chunks (DP = 4). A greedy parser that
        # commits at the central HH instead returns DP = 5.
        assert parse_motifs("HTHHTH") == (0, 2)

    def test_pure_python_and_featurizer_parsers_agree_exhaustively(self):
        from src.subjective_randomness.features import parse_motifs as feature_parse
        from src.subjective_randomness.model_families.common import (
            parse_motifs as family_parse,
        )

        for length in range(1, 9):
            for symbols in itertools.product("HT", repeat=length):
                sequence = "".join(symbols)
                assert family_parse(sequence) == feature_parse(sequence)

    def test_scores_are_the_difficulty_predictor(self):
        from src.subjective_randomness.model_families import falk_konold_dp

        # Score IS the Difficulty Predictor: harder to encode = more random.
        # Griffiths et al. (2018)'s worked example of the F&K parse:
        # HHTTHTHT -> (HH)(TT)(HTHT), DP = 4.
        assert falk_konold_dp.score_sequence("HHTTHTHT") == 4.0
        # Falk & Konold (1997, pp. 308-309) printed length-21 examples
        # (X -> H, O -> T): P(A)=.20 parses to five pure runs, DP = 5;
        # P(A)=.80 parses to five chunks, two alternating, DP = 7.
        assert falk_konold_dp.score_sequence("HHHHHHTTTHHTTTTTTTHHH") == 5.0
        assert falk_konold_dp.score_sequence("HHTHTHTHTHTTTHHTHTHTH") == 7.0

    def test_harder_to_encode_reads_as_more_random(self):
        from src.subjective_randomness.model_families import falk_konold_dp

        # Higher DP (harder encoding) -> more random-seeming (F&K's core claim).
        p = falk_konold_dp.predict_left(("HHTHTTHT", "HTHTHTHT"))
        assert p > 0.5
        p = falk_konold_dp.predict_left(("HHHHHHHH", "HHTHTTHT"))
        assert p < 0.5

    def test_dp_is_not_length_normalised(self):
        from src.subjective_randomness.model_families import falk_konold_dp

        # F&K never normalise DP by length: a short and a long sequence with
        # the same parse structure score identically.
        assert falk_konold_dp.score_sequence("HHTT") == falk_konold_dp.score_sequence(
            "HHHHTTTT"
        )


class TestMotifHMM:
    def test_hand_computed_probability_of_hh(self):
        from src.subjective_randomness.model_families import motif_hmm

        # Worked by hand from Eq. 9 at delta = alpha = 0.5:
        # P(HH) = (1/3)*(0.75/1.5) + (1/6)*(0.5/1.75) = 3/14.
        p = math.exp(motif_hmm.log_p_regular("HH", delta=0.5, alpha=0.5))
        assert p == pytest.approx(3.0 / 14.0, abs=1e-12)

    def test_forward_marginal_matches_brute_force_path_sum(self):
        from src.subjective_randomness.model_families import motif_hmm

        for delta, alpha in [(0.5, 0.366), (0.55, 0.21), (0.9, 0.05)]:
            for seq in ["H", "HT", "HHTH", "HTHTH", "TTTTT"]:
                brute = _brute_force_p_regular(seq, delta, alpha)
                forward = math.exp(motif_hmm.log_p_regular(seq, delta, alpha))
                assert forward == pytest.approx(brute, rel=1e-12), (seq, delta, alpha)

    def test_motif_process_is_a_proper_distribution_per_length(self):
        from src.subjective_randomness.model_families import motif_hmm

        # With row-normalised transitions and deterministic emissions, the
        # process defines a proper distribution over sequences of each length.
        total = sum(
            math.exp(motif_hmm.log_p_regular("".join(bits), delta=0.55, alpha=0.21))
            for bits in itertools.product("HT", repeat=6)
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_streaks_and_perfect_alternation_look_regular(self):
        from src.subjective_randomness.model_families import motif_hmm

        irregular = motif_hmm.score_sequence("HHTHTTHT")
        assert motif_hmm.score_sequence("HHHHHHHH") < irregular
        assert motif_hmm.score_sequence("HTHTHTHT") < irregular


class TestMotifStack:
    def test_paper_pattern_grammars_make_sequences_more_regular(self):
        from src.subjective_randomness.model_families import motif_stack

        asymmetric = "HHTHTTHT"
        assert motif_stack.score_sequence("HHHTTHHH") < motif_stack.score_sequence(
            asymmetric
        )
        assert motif_stack.score_sequence("TTTTHHHH") < motif_stack.score_sequence(
            asymmetric
        )
        assert motif_stack.score_sequence("HHTTHHTT") < motif_stack.score_sequence(
            asymmetric
        )


class TestOccurrenceProbability:
    def test_matches_hahn_warren_footnote_1_exactly(self):
        # H&W (2009) footnote 1: P(no run of 4 heads in n tosses) =
        # F4_{n+2}/2^n, given to 2 d.p. for n in {5,10,15,20,50,100,1000} as
        # {0.91, 0.75, 0.63, 0.52, 0.17, 0.03, 0.00}.
        from src.subjective_randomness.model_families.common import (
            occurrence_probability,
        )

        expected_nonoccurrence = {5: 0.91, 10: 0.75, 15: 0.63, 20: 0.52, 50: 0.17, 100: 0.03}
        for n, q in expected_nonoccurrence.items():
            assert 1.0 - occurrence_probability("HHHH", n) == pytest.approx(q, abs=0.005)

    def test_wait_times_match_hahn_warren_figure_3a(self):
        # Expected wait time = sum over n >= 0 of P(not yet occurred by n).
        # H&W p. 455-456 / Figure 3A: HHH=14, HHT=8, HHHH=30, HHHT=16, HTHT=20.
        from src.subjective_randomness.model_families.common import (
            occurrence_probability,
        )

        # Truncating the sum at n=400 leaves ~1e-5 of tail mass, so the
        # tolerance is 1e-3: loose enough for truncation, far tighter than
        # the integer gaps between the paper's wait times.
        def wait_time(pattern: str) -> float:
            return sum(1.0 - occurrence_probability(pattern, n) for n in range(400))

        assert wait_time("HHH") == pytest.approx(14.0, abs=1e-3)
        assert wait_time("HHT") == pytest.approx(8.0, abs=1e-3)
        assert wait_time("HHHH") == pytest.approx(30.0, abs=1e-3)
        assert wait_time("HHHT") == pytest.approx(16.0, abs=1e-3)
        assert wait_time("HTHT") == pytest.approx(20.0, abs=1e-3)

    def test_degenerate_windows(self):
        from src.subjective_randomness.model_families.common import (
            occurrence_probability,
        )

        # A pattern cannot occur in a window shorter than itself; a window
        # exactly its length contains it with probability (1/2)^k.
        assert occurrence_probability("HTH", 2) == 0.0
        assert occurrence_probability("HTH", 3) == pytest.approx(0.125)


class TestFiniteExperienceOccurrence:
    def test_rejects_cross_length_comparisons(self):
        from src.subjective_randomness.model_families import (
            finite_experience_occurrence,
        )

        with pytest.raises(ValueError, match="same-length"):
            finite_experience_occurrence.predict_left(("HT", "HTHT"))

    def test_score_uses_the_papers_focal_twenty_flip_stream(self):
        from src.subjective_randomness.model_families import (
            finite_experience_occurrence,
        )
        from src.subjective_randomness.model_families.common import (
            occurrence_probability,
        )

        assert finite_experience_occurrence.score_sequence("HHHT") == pytest.approx(
            math.log(occurrence_probability("HHHT", 20))
        )

    def test_streak_aversion_follows_occurrence_probability(self):
        from src.subjective_randomness.model_families import (
            finite_experience_occurrence,
        )

        # H&W p. 458: within 20 flips HHHT occurs with p ~ .75 vs ~ .48 for
        # HHHH, so HHHT should look more random.
        p = finite_experience_occurrence.predict_left(("HHHT", "HHHH"))
        assert p > 0.5

    def test_perfect_alternation_is_also_penalised(self):
        from src.subjective_randomness.model_families import (
            finite_experience_occurrence,
        )

        # HTHT (wait time 20) is rarer in finite experience than HHTT (16) —
        # the H&W signature that distinguishes this account from a pure
        # streak-aversion score like window_typicality's.
        assert finite_experience_occurrence.score_sequence(
            "HTHT"
        ) < finite_experience_occurrence.score_sequence("HHTT")

    def test_shorter_sequences_occur_more_often(self):
        from src.subjective_randomness.model_families import (
            finite_experience_occurrence,
        )

        # Documented auxiliary assumption (the paper never compares unequal
        # lengths): raw occurrence probability favours shorter strings.
        assert finite_experience_occurrence.score_sequence(
            "HT"
        ) > finite_experience_occurrence.score_sequence("HTHTHT")


class TestLocalRepresentativeness:
    def test_balance_is_aggregated_across_local_scales(self):
        from src.subjective_randomness.model_families.common import (
            multiscale_local_imbalance,
        )

        assert multiscale_local_imbalance(
            "HHHHTTTT"
        ) > multiscale_local_imbalance("HHTTHHTT")

    def test_obvious_periodicity_is_not_mistaken_for_local_balance(self):
        from src.subjective_randomness.model_families import local_representativeness

        patterned = "TTHHTTHH"
        irregular = "HHTHTTHT"
        assert local_representativeness.score_sequence(
            patterned
        ) < local_representativeness.score_sequence(irregular)

    def test_locality_separates_globally_balanced_sequences(self):
        from src.subjective_randomness.model_families import local_representativeness

        # K&T (1972, p. 435): balance must hold "locally in each of its
        # parts". HHHHTTTT and HHTTHHTT are both globally balanced; only the
        # local score separates them. With the alternation term switched off
        # this is a pure locality effect — invisible to prototype_similarity.
        params = dict(local_representativeness.DEFAULT_PARAMS, alt_weight=0.0)
        clumped = local_representativeness.score_sequence("HHHHTTTT", params)
        spread = local_representativeness.score_sequence("HHTTHHTT", params)
        assert spread > clumped

    def test_alternation_term_still_prefers_moderate_alternation(self):
        from src.subjective_randomness.model_families import local_representativeness

        # With default theta_alt ~ .65 a moderately alternating sequence beats
        # perfect alternation even though both are locally balanced.
        p = local_representativeness.predict_left(("HHTHTTHT", "HTHTHTHT"))
        assert p > 0.5
