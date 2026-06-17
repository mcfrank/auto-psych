"""Generate candidate stimulus pairs for EIG scoring.

Experiment 3 adds two new models beyond the experiment-2 set:
  - runs_test_model: Wald-Wolfowitz z-score (normalizes by sqrt(Var[R]) not n)
  - periodicity_salience: template-matching model using periodicity + imbalance

Strategies:
  1. Symmetric-deviation pairs: one streaky, one over-alternating, |p_alts-0.5| equal
     → discriminates asymmetric_alternation_prototype from inner_loop_model
  2. Mixed-length pairs: same deviation type at n=4 vs n=7/8
     → discriminates length_sensitive_prototype from inner_loop_model
  3. Compressibility pairs: periodic/long-run vs aperiodic at same imbalance
     → discriminates encoding_compressibility from prototype models
  4. Bayesian pairs: biased vs alternating (generator-type discrimination)
     → discriminates bayesian_diagnosticity from prototype models
  5. Classic pairs: maximum-deviation (streaky vs alternating) at every length
  6. Imbalanced-vs-balanced at matched run count: unbalanced sequences inflate
     Var[R] denominator in runs_test_model → discriminates it from inner_loop_model
  7. Periodicity-matched pairs: high vs low periodicity at same alts and imbalance
     → discriminates periodicity_salience from alternation-based models
"""

import itertools
import json
import math
import random


EXP_DIR = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/prototype_similarity/experiment3"

random.seed(42)


def count_alts(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def count_runs(seq):
    return count_alts(seq) + 1


def max_run(seq):
    return max(len(list(g)) for _, g in itertools.groupby(seq))


def periodicity(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    best = 0.0
    for period in range(1, n // 2 + 1):
        matches = sum(seq[i] == seq[i - period] for i in range(period, n))
        score = matches / (n - period)
        best = max(best, score)
    return best


def imbalance(seq):
    n = len(seq)
    h = seq.count("H")
    return abs(h - n / 2) / n


def p_alts(seq):
    n = len(seq)
    return count_alts(seq) / (n - 1) if n > 1 else 0.5


def wald_wolfowitz_var(seq):
    """Var[R] for Wald-Wolfowitz runs test = 2*h*t*(2*h*t - n) / (n^2*(n-1))."""
    n = len(seq)
    h = seq.count("H")
    t = n - h
    denom = n * n * (n - 1)
    if denom == 0:
        return 0.0
    return 2 * h * t * (2 * h * t - n) / denom


def all_seqs(length):
    return ["".join(s) for s in itertools.product("HT", repeat=length)]


candidates = []
seen = set()


def add_pair(a, b):
    key = tuple(sorted([a, b]))
    if key not in seen and a != b:
        seen.add(key)
        candidates.append({"sequence_a": a, "sequence_b": b})


# ── Strategy 1: Symmetric-deviation pairs ────────────────────────────────────
# Pairs where one sequence is streaky (p_alts < 0.5) and one is over-alternating
# (p_alts > 0.5) by the same deviation from 0.5.  Also require similar imbalance
# so the prototype model treats them as symmetric — any difference in model
# prediction must come from the asymmetric penalty.
strat1 = []
for length in range(5, 9):
    seqs = all_seqs(length)
    for sa in seqs:
        pa = p_alts(sa)
        if pa >= 0.5:
            continue
        dev_a = 0.5 - pa
        imb_a = imbalance(sa)
        for sb in seqs:
            pb = p_alts(sb)
            if pb <= 0.5:
                continue
            dev_b = pb - 0.5
            if abs(dev_a - dev_b) < 0.05 and abs(imb_a - imbalance(sb)) < 0.15:
                strat1.append((sa, sb))

random.shuffle(strat1)
for a, b in strat1[:60]:
    add_pair(a, b)

# ── Strategy 2: Mixed-length pairs ───────────────────────────────────────────
# Same p_alts and imbalance profile at a short vs long sequence.
# inner_loop_model: no length effect, predicts ~50/50
# length_sensitive_prototype: longer sequence penalised more by sqrt(n) scaling
strat2 = []
for (n_short, n_long) in [(4, 8), (5, 8), (4, 7)]:
    short_seqs = all_seqs(n_short)
    long_seqs = all_seqs(n_long)
    for sa in short_seqs:
        pa = p_alts(sa)
        imb_a = imbalance(sa)
        for sb in long_seqs:
            if abs(pa - p_alts(sb)) < 0.10 and abs(imb_a - imbalance(sb)) < 0.12:
                strat2.append((sa, sb))

random.shuffle(strat2)
for a, b in strat2[:50]:
    add_pair(a, b)

# ── Strategy 3: Compressibility pairs ────────────────────────────────────────
# High periodicity/long-run vs aperiodic/short-run, same imbalance and p_alts.
# encoding_compressibility penalises periodic structure and long runs separately
for length in range(5, 9):
    seqs = all_seqs(length)
    balanced = [s for s in seqs if imbalance(s) < 0.25]
    high_per = sorted(balanced, key=lambda s: -periodicity(s))[:5]
    low_per_lo_run = sorted(balanced, key=lambda s: (periodicity(s), max_run(s)))[:5]
    for a in high_per:
        for b in low_per_lo_run:
            if periodicity(a) > periodicity(b) + 0.25:
                add_pair(a, b)
    high_run = sorted(balanced, key=lambda s: -max_run(s))[:5]
    low_run = sorted(balanced, key=lambda s: max_run(s))[:5]
    for a in high_run:
        for b in low_run:
            if max_run(a) > max_run(b) + 1 and abs(p_alts(a) - p_alts(b)) < 0.2:
                add_pair(a, b)

# ── Strategy 4: Bayesian-diagnosticity pairs ─────────────────────────────────
# Biased vs alternating: Bayesian model assigns different non-random generator scores
for length in range(5, 9):
    seqs = all_seqs(length)
    biased = sorted([s for s in seqs if imbalance(s) > 0.4 and p_alts(s) < 0.45],
                    key=lambda s: -imbalance(s))[:4]
    alternating = sorted([s for s in seqs if p_alts(s) > 0.7 and imbalance(s) < 0.2],
                         key=lambda s: -p_alts(s))[:4]
    for a in biased:
        for b in alternating:
            add_pair(a, b)

# ── Strategy 5: Classic maximum-deviation pairs ───────────────────────────────
# Extreme streaky vs extreme alternating — always informative across all models
for length in range(4, 9):
    seqs = all_seqs(length)
    streaky = sorted(seqs, key=lambda s: (p_alts(s), -imbalance(s)))[:3]
    over_alt = sorted(seqs, key=lambda s: (-p_alts(s), imbalance(s)))[:3]
    for a in streaky:
        for b in over_alt:
            add_pair(a, b)

# ── Strategy 6: Imbalanced-vs-balanced at matched alternation count ───────────
# runs_test_model normalizes by sqrt(Var[R]); Var[R] → 0 for unbalanced seqs
# → z-score inflated for unbalanced sequences relative to inner_loop_model.
# Pairs: one highly unbalanced (imbalance > 0.4), one balanced (imbalance < 0.15),
# with similar raw alternation counts → models should diverge on which looks more random.
strat6 = []
for length in range(5, 9):
    seqs = all_seqs(length)
    unbalanced = [s for s in seqs if imbalance(s) > 0.38]
    balanced = [s for s in seqs if imbalance(s) < 0.15]
    for sa in unbalanced:
        alts_a = count_alts(sa)
        var_a = wald_wolfowitz_var(sa)
        for sb in balanced:
            alts_b = count_alts(sb)
            var_b = wald_wolfowitz_var(sb)
            # Match raw alternation count but differ substantially in Var[R]
            if abs(alts_a - alts_b) <= 1 and var_b > var_a + 0.02:
                strat6.append((sa, sb))

random.shuffle(strat6)
for a, b in strat6[:50]:
    add_pair(a, b)

# ── Strategy 7: Periodicity-matched pairs ────────────────────────────────────
# High vs low periodicity at matched alternation rate and imbalance.
# Prototype/inner_loop models see ~equal features → predict ~50/50.
# periodicity_salience predicts the high-periodicity sequence looks less random.
strat7 = []
for length in range(5, 9):
    seqs = all_seqs(length)
    for sa in seqs:
        per_a = periodicity(sa)
        if per_a < 0.7:
            continue
        pa = p_alts(sa)
        imb_a = imbalance(sa)
        for sb in seqs:
            per_b = periodicity(sb)
            if per_b > 0.4:
                continue
            # Match alternation and imbalance so other models see equivalent stimuli
            if abs(pa - p_alts(sb)) < 0.12 and abs(imb_a - imbalance(sb)) < 0.15:
                strat7.append((sa, sb))

random.shuffle(strat7)
for a, b in strat7[:50]:
    add_pair(a, b)


print(f"Generated {len(candidates)} candidate pairs")
out_path = f"{EXP_DIR}/design/candidates.json"
with open(out_path, "w") as fh:
    json.dump(candidates, fh, indent=2)
print(f"Written to {out_path}")
