"""Generate candidate stimulus pairs for EIG scoring.

Experiment 2 adds two new models beyond the experiment-1 set:
  - length_sensitive_prototype: multiplies each penalty by sqrt(n)
  - asymmetric_alternation_prototype: streak_k scales below-prototype penalty

Strategies:
  1. Symmetric-deviation pairs: one streaky, one over-alternating, |p_alts-0.5| equal
     → discriminates asymmetric_alternation_prototype from inner_loop_model
  2. Mixed-length pairs: same deviation type at n=4 vs n=7/8
     → discriminates length_sensitive_prototype from inner_loop_model
  3. Compressibility pairs: periodic/long-run vs aperiodic at same imbalance
     → discriminates encoding_compressibility from prototype models
  4. Bayesian pairs: biased vs alternating (generator-type discrimination)
     → discriminates bayesian_diagnosticity from prototype models
  5. Classic pairs: maximum-deviation (all-H vs all-alt) at every length
"""

import itertools
import json
import random


EXP_DIR = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/prototype_similarity/experiment2"

random.seed(42)


def count_alts(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


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
for a, b in strat1[:70]:
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
for a, b in strat2[:60]:
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


print(f"Generated {len(candidates)} candidate pairs")
out_path = f"{EXP_DIR}/design/candidates.json"
with open(out_path, "w") as fh:
    json.dump(candidates, fh, indent=2)
print(f"Written to {out_path}")
