"""Generate candidate stimulus pairs for EIG scoring.

5 competing models:
  - fair_coin_run_baseline: max_run excess relative to kappa * log2(n) — new in exp3
  - encoding_compressibility: weighted penalty of max_run_norm + periodicity + imbalance
  - bayesian_diagnosticity: log-ratio of fair vs {alternating, streaky, biased} generators
  - inner_loop_model: Bayesian diagnosticity with learned streak_switch_prob
  - alternation_prototype: L1 distance of p_alts to learned prototype theta_alt

Key discriminating pair categories:
  1. FCR vs BD/inner — highly-alternating (low max_run, FCR: random) vs balanced-moderate (BD: random)
       FCR cares only about max_run relative to log2(n); BD penalizes alternating generator match
  2. FCR vs EC via log-vs-linear length normalization — same log-excess, different linear-norm
       e.g., n=4 max_run=2 vs n=8 max_run=3 (both have excess~0); EC says n=8 is less random
  3. FCR vs EC via imbalance — same max_run but one sequence is imbalanced
       FCR indifferent to imbalance; EC penalizes it
  4. FCR vs EC via periodicity — periodic vs aperiodic, similar max_run
       FCR indifferent to periodicity; EC penalizes it
  5. AP vs BD — medium-alternating vs fair-balanced (carried from exp2)
  6. EC vs BD/FCR — streaky (long runs, few alts) vs alternating
"""

import itertools
import json
import math
import os
import random


EXP_DIR = os.path.dirname(__file__)


def _alternations(s):
    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])


def _max_run(s):
    m, cur, prev = 1, 1, ""
    for c in s:
        cur = cur + 1 if c == prev else 1
        prev = c
        m = max(m, cur)
    return m


def _periodicity(s):
    n = len(s)
    if n <= 2:
        return 0.0
    best = 0.5
    for p in range(1, (n // 2) + 1):
        template = s[:p]
        matches = sum(1 for i, c in enumerate(s) if c == template[i % p])
        best = max(best, matches / n)
    return max(0.0, min(1.0, 2.0 * (best - 0.5)))


def _features(s):
    n = len(s)
    h = s.count("H")
    alts = _alternations(s)
    mx = _max_run(s)
    return {
        "n": n,
        "h": h,
        "alts": alts,
        "max_run": mx,
        "p_alts": alts / (n - 1) if n > 1 else 0.0,
        "max_run_norm": (mx - 1) / (n - 1) if n > 1 else 0.0,
        "imbalance": 2.0 * abs(h / n - 0.5),
        "periodicity": _periodicity(s),
        "log_excess": mx - math.log2(max(n, 2)),
    }


def _enumerate_seqs(min_len=4, max_len=8):
    seqs = []
    for length in range(min_len, max_len + 1):
        for bits in itertools.product("HT", repeat=length):
            seqs.append("".join(bits))
    return seqs


def main():
    random.seed(42)
    seqs = _enumerate_seqs(4, 8)
    by_seq = {s: _features(s) for s in seqs}

    # ---- classify sequences by archetype ----------------------------------------

    # Highly alternating: p_alts > 0.80, max_run <= 2
    # FCR: very low log-excess (looks random); BD: diagnostic of alternating generator (not random)
    high_alt = [
        s for s, f in by_seq.items()
        if f["p_alts"] > 0.80 and f["max_run"] <= 2
    ]

    # Balanced moderate: p_alts in [0.30, 0.60], balanced, moderate run
    # BD and inner_loop prefer (fair-coin-like); FCR also comfortable (low excess)
    balanced_moderate = [
        s for s, f in by_seq.items()
        if 0.30 <= f["p_alts"] <= 0.60 and f["imbalance"] < 0.15 and 2 <= f["max_run"] <= 3
    ]

    # Cross-length same-log-excess (low): n=4, max_run=2 vs n=8, max_run=3
    # FCR: both have excess ~0 (2-log2(4)=0, 3-log2(8)=0) — indifferent
    # EC: n=4 has max_run_norm=1/3=0.33; n=8 has max_run_norm=2/7=0.29 — EC prefers n=8
    short_low_excess = [
        s for s, f in by_seq.items()
        if f["n"] == 4 and f["max_run"] == 2 and f["imbalance"] < 0.35
    ]
    long_low_excess = [
        s for s, f in by_seq.items()
        if f["n"] == 8 and f["max_run"] == 3 and f["imbalance"] < 0.35
    ]

    # Cross-length same-log-excess (mid): n=4, max_run=3 vs n=8, max_run=4
    # FCR: both have excess ~1 — indifferent
    # EC: n=4 has max_run_norm=2/3=0.67; n=8 has max_run_norm=3/7=0.43 — EC prefers n=8
    short_mid_excess = [
        s for s, f in by_seq.items()
        if f["n"] == 4 and f["max_run"] == 3 and f["imbalance"] < 0.35
    ]
    long_mid_excess = [
        s for s, f in by_seq.items()
        if f["n"] == 8 and f["max_run"] == 4 and f["imbalance"] < 0.35
    ]

    # Imbalanced short run: max_run <= 2 but high imbalance (FCR ignores imbalance; EC penalizes)
    imbalanced_short_run = [
        s for s, f in by_seq.items()
        if f["max_run"] <= 2 and f["imbalance"] > 0.35 and f["n"] >= 6
    ]

    # Balanced short run: max_run <= 2, very balanced (FCR and EC both prefer)
    balanced_short_run = [
        s for s, f in by_seq.items()
        if f["max_run"] <= 2 and f["imbalance"] < 0.10 and f["n"] >= 6
    ]

    # Periodic sequences: high periodicity, short run (FCR ignores periodicity; EC penalizes)
    periodic = [
        s for s, f in by_seq.items()
        if f["periodicity"] > 0.40 and f["max_run"] <= 3 and f["n"] >= 6
    ]

    # Aperiodic moderate: similar length and run to periodic group, but aperiodic
    aperiodic_moderate = [
        s for s, f in by_seq.items()
        if f["periodicity"] < 0.10 and 2 <= f["max_run"] <= 3
        and f["imbalance"] < 0.20 and 0.30 <= f["p_alts"] <= 0.60 and f["n"] >= 6
    ]

    # Medium alternating: AP's ideal range (close to theta_alt ~0.65-0.75)
    medium_alt = [
        s for s, f in by_seq.items()
        if 0.55 <= f["p_alts"] <= 0.82 and f["imbalance"] < 0.15 and f["n"] >= 5
    ]

    # Fair balanced: BD's ideal (balanced h, moderate alts)
    fair_balanced = [
        s for s, f in by_seq.items()
        if 0.35 <= f["p_alts"] <= 0.55 and f["imbalance"] < 0.10 and f["max_run"] <= 3
    ]

    # Streaky: long runs, few alternations (EC and FCR penalize; BD penalizes via streaky generator)
    streaky = [
        s for s, f in by_seq.items()
        if f["max_run"] >= 4 and f["p_alts"] < 0.30 and f["n"] >= 6
    ]

    # ---- generate pairs from each category ----------------------------------------

    pairs = []
    seen = set()

    def _add_pair(a, b):
        key = (min(a, b), max(a, b))
        if key not in seen and a != b:
            seen.add(key)
            pairs.append({"sequence_a": a, "sequence_b": b})

    def _sample_pairs(list_a, list_b, n=30):
        a_samp = random.sample(list_a, min(len(list_a), 60))
        b_samp = random.sample(list_b, min(len(list_b), 60))
        added = 0
        for a, b in itertools.product(a_samp, b_samp):
            if added >= n:
                break
            _add_pair(a, b)
            added += 1

    # Cat 1: FCR vs BD — highly-alt (FCR: random) vs balanced-moderate (BD: random)
    _sample_pairs(high_alt, balanced_moderate, 40)

    # Cat 2a: FCR vs EC via log-normalization (low excess cross-length)
    _sample_pairs(short_low_excess, long_low_excess, 30)

    # Cat 2b: FCR vs EC via log-normalization (mid excess cross-length)
    _sample_pairs(short_mid_excess, long_mid_excess, 30)

    # Cat 3: FCR vs EC via imbalance (same short run, different imbalance)
    _sample_pairs(imbalanced_short_run, balanced_short_run, 30)

    # Cat 4: FCR vs EC via periodicity (periodic vs aperiodic, similar run)
    _sample_pairs(periodic, aperiodic_moderate, 25)

    # Cat 5: AP vs BD — medium-alt vs fair-balanced (carried from exp2)
    _sample_pairs(medium_alt, fair_balanced, 30)

    # Cat 6: EC/FCR vs BD — streaky vs alternating
    _sample_pairs(streaky, high_alt, 25)

    random.shuffle(pairs)

    design_dir = os.path.join(EXP_DIR, "design")
    os.makedirs(design_dir, exist_ok=True)

    out_path = os.path.join(design_dir, "candidates.json")
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} candidate pairs to {out_path}")


if __name__ == "__main__":
    main()
