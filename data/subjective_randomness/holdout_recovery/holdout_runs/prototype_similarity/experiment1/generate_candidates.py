"""Generate candidate stimulus pairs for EIG scoring.

Targets pairs that discriminate between bayesian_diagnosticity (uses n/h/alts)
and encoding_compressibility (uses max_run_norm/periodicity/imbalance).
"""

import itertools
import json
import math


EXP_DIR = "/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery/holdout_runs/prototype_similarity/experiment1"


def count_alts(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])


def max_run(seq):
    return max(len(list(g)) for _, g in itertools.groupby(seq))


def periodicity(seq):
    """Fraction of positions matching a shifted version at the best period."""
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


def features(seq):
    n = len(seq)
    h = seq.count("H")
    alts = count_alts(seq)
    mr = max_run(seq)
    mr_norm = mr / n
    per = periodicity(seq)
    imb = imbalance(seq)
    return dict(n=n, h=h, alts=alts, max_run=mr, max_run_norm=mr_norm, periodicity=per, imbalance=imb)


def all_seqs(length):
    return ["".join(s) for s in itertools.product("HT", repeat=length)]


candidates = []
seen = set()


def add_pair(a, b):
    key = tuple(sorted([a, b]))
    if key not in seen and a != b:
        seen.add(key)
        candidates.append({"sequence_a": a, "sequence_b": b})


# --- Strategy 1: High-alts vs high-max_run (same length, same head count) ---
# These pairs directly target the discriminating axis:
#   bayesian dislikes high alts (alternating generator)
#   encoding dislikes high max_run
for length in range(4, 9):
    seqs = all_seqs(length)
    by_headcount = {}
    for s in seqs:
        h = s.count("H")
        by_headcount.setdefault(h, []).append(s)

    for h, group in by_headcount.items():
        high_alts = sorted(group, key=lambda s: -count_alts(s))[:4]
        high_run = sorted(group, key=lambda s: -max_run(s))[:4]
        for a in high_alts:
            for b in high_run:
                add_pair(a, b)

# --- Strategy 2: Biased (high h) vs alternating (same length) ---
# bayesian dislikes both; encoding dislikes imbalance more than periodicity
for length in range(4, 9):
    seqs = all_seqs(length)
    # Very biased toward H
    biased_h = [s for s in seqs if s.count("H") / length >= 0.75]
    # Very biased toward T
    biased_t = [s for s in seqs if s.count("H") / length <= 0.25]
    # Alternating (high alts, balanced)
    alternating = [s for s in seqs if count_alts(s) >= length - 2 and s.count("H") == length // 2]
    for b in biased_h[:8]:
        for a in alternating[:6]:
            add_pair(a, b)
    for b in biased_t[:8]:
        for a in alternating[:6]:
            add_pair(a, b)

# --- Strategy 3: Periodic but short-run vs. aperiodic but long-run ---
# encoding's periodicity vs max_run tradeoff within the model
# bayesian_diagnosticity would be indifferent to this distinction
for length in range(5, 9):
    seqs = all_seqs(length)
    balanced = [s for s in seqs if abs(s.count("H") - length / 2) <= 1]
    # High periodicity, low max_run: HTHTTH-style
    high_per = sorted(balanced, key=lambda s: (-periodicity(s), max_run(s)))[:10]
    # Low periodicity, high max_run: HHHTT-style
    high_run = sorted(balanced, key=lambda s: (-max_run(s), periodicity(s)))[:10]
    for a in high_per:
        for b in high_run:
            if periodicity(a) > periodicity(b) and max_run(b) > max_run(a):
                add_pair(a, b)

# --- Strategy 4: Imbalanced+short-run vs balanced+long-run ---
# encoding: imbalance penalty vs max_run penalty tradeoff
# bayesian: imbalance drives biased-generator score
for length in range(4, 9):
    seqs = all_seqs(length)
    imb_short = [s for s in seqs if imbalance(s) > 0.3 and max_run(s) <= 2]
    bal_long = [s for s in seqs if imbalance(s) < 0.15 and max_run(s) >= length // 2]
    for a in imb_short[:8]:
        for b in bal_long[:8]:
            add_pair(a, b)

# --- Strategy 5: Diverse random-looking pairs from length 6-8 ---
# Sample pairs where both sequences look "intermediate" to one model but
# differ markedly on the other model's metric
for length in range(6, 9):
    seqs = all_seqs(length)
    balanced = [s for s in seqs if abs(s.count("H") - length / 2) <= 1]
    # Sort by |alts - n/2|: sequences with very different alternation counts
    by_alts = sorted(balanced, key=lambda s: count_alts(s))
    low_alts = by_alts[:8]
    high_alts = by_alts[-8:]
    for a in high_alts:
        for b in low_alts:
            add_pair(a, b)

# --- Strategy 6: Length-matched pairs spanning the bayesian model's key contrasts ---
# The bayesian model scores fair vs. alternating/biased/streaky.
# Pairs that pit two non-random sequence types against each other are most informative.
for length in range(5, 9):
    seqs = all_seqs(length)
    # Near-perfect alternators
    near_alt = [s for s in seqs if count_alts(s) >= length - 2]
    # Near-perfect streamers (long run of H or T)
    near_streak = [s for s in seqs if max_run(s) >= length - 1]
    for a in near_alt[:6]:
        for b in near_streak[:6]:
            add_pair(a, b)


print(f"Total candidates: {len(candidates)}")

# Trim to ~200 by keeping pairs most likely to be discriminating:
# Prioritize pairs where max_run differs a lot between sequences
def discriminating_score(pair):
    a, b = pair["sequence_a"], pair["sequence_b"]
    fa, fb = features(a), features(b)
    delta_alts = abs(fa["alts"] - fb["alts"]) / max(fa["n"], fb["n"])
    delta_run = abs(fa["max_run_norm"] - fb["max_run_norm"])
    delta_per = abs(fa["periodicity"] - fb["periodicity"])
    delta_imb = abs(fa["imbalance"] - fb["imbalance"])
    # High score = likely to tease apart the two models
    return delta_alts + delta_run + delta_per + delta_imb


candidates.sort(key=discriminating_score, reverse=True)
candidates = candidates[:200]
print(f"After trim: {len(candidates)}")

out_path = f"{EXP_DIR}/design/candidates.json"
with open(out_path, "w") as f:
    json.dump(candidates, f, indent=2)
print(f"Written to {out_path}")
