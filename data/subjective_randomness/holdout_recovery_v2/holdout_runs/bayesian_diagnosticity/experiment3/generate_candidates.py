#!/usr/bin/env python3
"""Generate candidate stimulus pairs for Bayesian Diagnosticity experiment 3.

Seven cognitive models compete; each has a distinct combination of features:
  - prototype_similarity: L1 penalty on p_alts deviation + imbalance
  - encoding_compressibility: penalty on max_run_norm, periodicity, imbalance
  - length_sensitive_alternation: count-scale quadratic alternation deviation
  - bayesian_markov_fairness: log-Bayes-factor for fair vs biased Markov chain
  - run_length_prototype: prefer intermediate max_run_norm (~1/3 of length)
  - length_sensitive_2d_prototype: length-scaled Gaussian 2D prototype
  - inner_loop_model: Gaussian 2D prototype on proportion scale

Pairs are chosen to span regions of feature space where models disagree:
  1. High alternation (max_run=1) vs intermediate max run (~n/3): discriminates
     run_length_prototype from alternation-based models.
  2. Periodic vs aperiodic with similar alternation rate: discriminates
     encoding_compressibility from the rest.
  3. Cross-length pairs with the same p_alts: discriminates length-sensitive
     models (length_sensitive_alternation, length_sensitive_2d_prototype,
     bayesian_markov_fairness) from proportion-scale models.
  4. Concentrated deviation (one dimension) vs distributed deviation (both
     dimensions): discriminates L1-based (prototype_similarity) from
     L2-based (inner_loop_model) penalty shapes.
  5. Count-scale vs proportion-scale: cross-length pairs where alternation
     count is matched but proportion differs.
"""

import json
from pathlib import Path


EXP_DIR = Path(__file__).parent
DESIGN_DIR = EXP_DIR / "design"
DESIGN_DIR.mkdir(exist_ok=True)
OUTPUT = DESIGN_DIR / "candidates.json"


def max_run_len(seq: str) -> int:
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def featurize(seq: str) -> dict:
    n = len(seq)
    h = seq.count("H")
    alts = sum(seq[i] != seq[i + 1] for i in range(n - 1))
    p_alts = alts / (n - 1) if n > 1 else 0.0
    mr = max_run_len(seq)
    return {
        "n": n,
        "h": h,
        "p": h / n,
        "alts": alts,
        "p_alts": p_alts,
        "max_run": mr,
        "max_run_norm": mr / n,
        "imbalance": abs(h - n / 2) / (n / 2) if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Curated sequence pool — grouped by design rationale
# ---------------------------------------------------------------------------

# Group A: perfect alternation (max_run=1) at each length
# EIG expects run_length_prototype to disagree here because max_run_norm << ideal
PERFECT_ALT = [
    "HTHT",       # n=4, p_alts=1.0
    "THTH",
    "HTHTH",      # n=5
    "THTHT",
    "HTHTHT",     # n=6
    "THTHTH",
    "HTHTHTH",    # n=7
    "THTHTTH",    # n=7 but with one break — wait compute by hand
    "HTHTHTHT",   # n=8, p_alts=1.0
    "THTHTTH",    # length 7 — let me verify: T,H,T,H,T,T,H
]

# Group B: intermediate max run (~1/3 of length), balanced
# run_length_prototype prefers these; alternation-based models may not
INTERMEDIATE_MAX_RUN = [
    "HHTT",       # n=4, max_run=2=0.5 — slightly above ideal for n=4
    "HTTH",       # n=4, max_run=2
    "THHT",       # n=4, max_run=2
    "HHTTH",      # n=5, max_run=2=0.4
    "TTHHT",      # n=5, max_run=2=0.4
    "HHTTHT",     # n=6, max_run=2=0.33 — near ideal
    "HTTHHT",     # n=6, max_run=2=0.33
    "TTHHT",      # dup check... n=5
    "HHTTHTTH",   # too long? H,H,T,T,H,T,T,H = 8 chars, max_run=2=0.25
    "HTTHHTTH",   # n=8, max_run=2=0.25
    "HTHTTHH",    # n=7: H,T,H,T,T,H,H → max_run=2
    "THHTTHH",    # n=7: T,H,H,T,T,H,H → max_run=2
]

# Group C: low alternation, balanced (runs dominate)
LOW_ALT_BALANCED = [
    "HHTT",       # n=4, p_alts=0.33
    "TTHH",       # n=4
    "HHHTTT",     # n=6, p_alts=0.2, max_run=3=0.5
    "TTTHHH",     # n=6
    "HHTTHHTT",   # n=8, p_alts=0.43, PERIODIC (period 4), max_run=2
    "HHHHTTTT",   # n=8, p_alts=0.14, max_run=4=0.5
    "TTTTHHH",    # n=7, p_alts=0.17
    "HHHTTTTH",   # n=8: H,H,H,T,T,T,T,H → max_run=4
    "HHTTTTH",    # n=7: H,H,T,T,T,T,H → max_run=4
]

# Group D: periodic sequences (period 2 or 4) — encoding_compressibility penalizes these
PERIODIC = [
    "HTHT",       # period 2, n=4
    "THTHTH",     # period 2, n=6
    "HTHTHTHT",   # period 2, n=8
    "HHTTHHTT",   # period 4, n=8
    "TTHHTTHHTT", # too long (10)
    "TTHHTTHH",   # period 4, n=8
    "HHTTH",      # period-ish n=5 — H,H,T,T,H
]

# Group E: unbalanced sequences
UNBALANCED = [
    "HHHT",       # n=4, h=3
    "HHTH",       # n=4, h=3
    "HHHTH",      # n=5, h=4
    "HHTHTH",     # n=6: H,H,T,H,T,H → h=4, p_alts=4/5=0.8
    "HHHTHH",     # n=6: h=5, low alt
    "HTHTHHTT",   # n=8: H,T,H,T,H,H,T,T → h=5, p_alts=5/7≈0.71
    "HHHHTTTH",   # n=8: h=5, low alt, max_run=4
    "HTTTHTHH",   # n=8: H,T,T,T,H,T,H,H → h=4... wait let me count: H,T,T,T,H,T,H,H = 8, h=4... not unbalanced
    "HHHTTTT",    # n=7: h=3, p_alts=2/6=0.33
    "TTTTHH",     # n=6: h=2, unbalanced
    "HHTHTTH",    # n=7: H,H,T,H,T,T,H → h=4, p_alts=4/6=0.67
]

# Group F: medium alternation, balanced — near prototype ideal
MEDIUM_ALT_BALANCED = [
    "HTTH",       # n=4, p_alts=0.67
    "THHT",       # n=4
    "HHTTH",      # n=5, p_alts=0.50
    "HTTHT",      # n=5: H,T,T,H,T → p_alts=3/4=0.75, h=2... not quite balanced
    "HTTHHT",     # n=6, p_alts=0.6, balanced
    "HTHTTH",     # n=6: H,T,H,T,T,H → p_alts=4/5=0.8
    "THHTTHT",    # n=7: T,H,H,T,T,H,T → h=3, p_alts=4/6=0.67
    "HHTTHTH",    # n=7: H,H,T,T,H,T,H → h=4, p_alts=4/6=0.67
    "HTTHHTTH",   # n=8, p_alts=4/7=0.57, balanced
    "HTTHTTHH",   # n=8: H,T,T,H,T,T,H,H → h=4, p_alts=4/7=0.57
]

# Combine all unique sequences (removing duplicates)
all_seqs = set()
for group in [
    PERFECT_ALT, INTERMEDIATE_MAX_RUN, LOW_ALT_BALANCED,
    PERIODIC, UNBALANCED, MEDIUM_ALT_BALANCED
]:
    for s in group:
        if 4 <= len(s) <= 8:
            all_seqs.add(s)

seqs = sorted(all_seqs)
print(f"Total unique sequences: {len(seqs)}")

# Print feature summary for debugging
for s in sorted(seqs, key=len):
    f = featurize(s)
    print(
        f"  {s:<12} n={f['n']} p_alts={f['p_alts']:.2f} "
        f"max_run_norm={f['max_run_norm']:.2f} imb={f['imbalance']:.2f}"
    )

# ---------------------------------------------------------------------------
# Generate all pairs, then add targeted cross-length pairs
# ---------------------------------------------------------------------------

# All within-set pairs
pairs = []
seq_list = sorted(seqs, key=lambda s: (len(s), s))
for i, a in enumerate(seq_list):
    for b in seq_list[i + 1 :]:
        if a != b:
            pairs.append({"sequence_a": a, "sequence_b": b})

print(f"\nAll pairs: {len(pairs)}")

# ---------------------------------------------------------------------------
# Targeted cross-length pairs for discriminating length-sensitive models
# ---------------------------------------------------------------------------
# Same p_alts but different lengths → length_sensitive models score differently
cross_length = [
    # Perfect alternation across lengths
    ("HTHT", "HTHTHT"),      # n=4 vs n=6, both p_alts=1.0
    ("HTHT", "HTHTHTHT"),    # n=4 vs n=8
    ("HTHTHT", "HTHTHTHT"),  # n=6 vs n=8
    ("THTH", "THTHTTH"),     # n=4 vs n=7 (check if ththtth has right alts)

    # Low alternation across lengths — count scale vs proportion scale
    ("HHTT", "HHHTTT"),      # n=4 vs n=6 both p_alts~0.33
    ("HHTT", "HHHHTTTT"),    # n=4 vs n=8 — p_alts differ but counts matched
    ("HTTH", "HTTHHTTH"),    # n=4 vs n=8 — similar p_alts

    # Medium alternation across lengths
    ("HTTH", "HTHTTH"),      # n=4 vs n=6 — different p_alts
    ("HHTTH", "HHTTHTH"),    # n=5 vs n=7

    # Same alts COUNT but different lengths (discriminates count vs proportion models)
    ("HHTT", "HHTTH"),       # alts=1, n=4 vs alts=2, n=5 — same count but different n
    ("HHTT", "HHTTHT"),      # alts=1 vs alts=3, n=4 vs n=6
    ("HTHT", "HTHTHTH"),     # perfect alt across lengths

    # High alternation short vs medium alternation long
    ("HTHT", "HHTTHTTH"),    # n=4 p_alts=1.0 vs n=8 p_alts=0.57
    ("HTHTH", "HTTHHTTH"),   # n=5 vs n=8
]

for a, b in cross_length:
    if 4 <= len(a) <= 8 and 4 <= len(b) <= 8:
        pairs.append({"sequence_a": a, "sequence_b": b})

# ---------------------------------------------------------------------------
# Targeted pairs for run_length_prototype discrimination
# ---------------------------------------------------------------------------
# Pairs where one has max_run near ideal (~1/3 of length) and the other has
# extreme max_run (either 1 or close to n)
run_length_pairs = [
    ("HTHTHT", "HHTTHT"),     # p_alts=1.0,max_run=1 vs p_alts=0.6,max_run=2
    ("HTHTHTHT", "HTTHHTTH"), # perfect alt vs moderate alt, max_run=2
    ("HTHTHTHT", "HHTTHHTT"), # perfect alt vs periodic, max_run=2
    ("HTHTHTHT", "HHHHTTTT"), # perfect alt vs no alt, max_run=4
    ("HTHTHTH", "HTHTTHH"),   # n=7 perfect alt vs max_run=2
    ("HTHT", "HHTT"),         # n=4 perfect alt vs max_run=2
    ("HTHT", "HTTH"),         # n=4 perfect alt vs max_run=2 medium alt
    ("HTHTH", "HHTTH"),       # n=5 perfect alt vs max_run=2
    ("HTHTH", "HHHTTT"),      # n=5 vs n=6 (cross-length version)
    ("HTHTHT", "HHHTTT"),     # n=6 perfect alt vs max_run=3
    ("HTHTHTHT", "HHTTTTH"),  # n=8 vs n=7 — won't add (cross-length...)
]
for a, b in run_length_pairs:
    if 4 <= len(a) <= 8 and 4 <= len(b) <= 8:
        # only add if not already present
        if {"sequence_a": a, "sequence_b": b} not in pairs:
            pairs.append({"sequence_a": a, "sequence_b": b})

# ---------------------------------------------------------------------------
# Targeted pairs for encoding_compressibility (periodicity) discrimination
# ---------------------------------------------------------------------------
periodicity_pairs = [
    ("HHTTHHTT", "HTTHHTTH"),  # periodic vs similar but aperiodic, both n=8
    ("HHTTHHTT", "HTHTHHTT"),  # periodic vs high-alt unbalanced
    ("HTHTHTHT", "HTTHHTTH"),  # period-2 alt vs aperiodic moderate
    ("THTHTH", "HTTHHT"),      # n=6 perfect alt (periodic) vs aperiodic
    ("HTHTH", "HHTTH"),        # n=5 perfect alt vs aperiodic
    ("HHTTHHTT", "HTTHTTHH"),  # periodic vs aperiodic, similar features
    ("HHTTHHTT", "HHHHTTTT"),  # periodic vs one big run
]
for a, b in periodicity_pairs:
    if 4 <= len(a) <= 8 and 4 <= len(b) <= 8:
        if {"sequence_a": a, "sequence_b": b} not in pairs:
            pairs.append({"sequence_a": a, "sequence_b": b})

# Deduplicate (including reverse pairs)
seen = set()
unique_pairs = []
for p in pairs:
    key = tuple(sorted([p["sequence_a"], p["sequence_b"]]))
    if key not in seen and p["sequence_a"] != p["sequence_b"]:
        seen.add(key)
        unique_pairs.append(p)

print(f"Unique pairs after dedup: {len(unique_pairs)}")

# Trim to ≤250 if needed
if len(unique_pairs) > 250:
    # Keep the first 250 — the all-pairs block is already sorted, so
    # targeted pairs near the end get priority trimming. Shuffle first
    # to avoid losing targeted pairs; use a fixed seed for reproducibility.
    import random
    rng = random.Random(42)
    rng.shuffle(unique_pairs)
    unique_pairs = unique_pairs[:250]
    print(f"Trimmed to {len(unique_pairs)} pairs")

OUTPUT.write_text(json.dumps(unique_pairs, indent=2))
print(f"\nWrote {len(unique_pairs)} candidates to {OUTPUT}")
