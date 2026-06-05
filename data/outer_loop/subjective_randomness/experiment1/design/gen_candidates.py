"""Generate candidate stimulus pairs for the subjective_randomness experiment.

Creates all ordered (A, B) pairs from a hand-curated set of sequences that span
the feature space of the three cognitive models: alternation_bias (p_alts),
runs_penalty (max_run), and bayesian_fair_coin (h/n, n).
"""
import itertools
import json
from pathlib import Path

# fmt: off
# 24 sequences covering lengths 3, 4, 6, 8.
# Annotations: n=length, h=heads, p_alts=alts/(n-1), max_run=longest run
SEQUENCES = [
    # --- length 3 ---
    "HTH",      # n=3, h=2, p_alts=1.00, max_run=1  (perfectly alternating)
    "HHT",      # n=3, h=2, p_alts=0.50, max_run=2
    "HHH",      # n=3, h=3, p_alts=0.00, max_run=3  (all heads)
    "TTT",      # n=3, h=0, p_alts=0.00, max_run=3  (all tails)
    # --- length 4 ---
    "HTHT",     # n=4, h=2, p_alts=1.00, max_run=1  (perfectly alternating)
    "THTH",     # n=4, h=2, p_alts=1.00, max_run=1  (perfectly alternating, T-start)
    "HHTT",     # n=4, h=2, p_alts=0.33, max_run=2
    "HHHT",     # n=4, h=3, p_alts=0.33, max_run=3
    "HTTT",     # n=4, h=1, p_alts=0.33, max_run=3
    "HHHH",     # n=4, h=4, p_alts=0.00, max_run=4  (all heads)
    # --- length 6 ---
    "HTHTHT",   # n=6, h=3, p_alts=1.00, max_run=1  (perfectly alternating)
    "THTHTH",   # n=6, h=3, p_alts=1.00, max_run=1  (perfectly alternating, T-start)
    "HHTHTT",   # n=6, h=3, p_alts=0.60, max_run=2
    "HHHTTT",   # n=6, h=3, p_alts=0.20, max_run=3
    "HHTTHH",   # n=6, h=4, p_alts=0.40, max_run=2
    "HTTTTH",   # n=6, h=2, p_alts=0.40, max_run=4  (long T-run)
    "HHHHHH",   # n=6, h=6, p_alts=0.00, max_run=6  (all heads)
    "TTTTTT",   # n=6, h=0, p_alts=0.00, max_run=6  (all tails)
    # --- length 8 ---
    "HTHTHTHT", # n=8, h=4, p_alts=1.00, max_run=1  (perfectly alternating)
    "THTHTHTH", # n=8, h=4, p_alts=1.00, max_run=1  (perfectly alternating, T-start)
    "HHTTHHTT", # n=8, h=4, p_alts=0.29, max_run=2
    "HTTTTTTH", # n=8, h=2, p_alts=0.29, max_run=6  (long T-run)
    "HHHHHHTH", # n=8, h=7, p_alts=0.14, max_run=6  (long H-run)
    "TTTTTTTT", # n=8, h=0, p_alts=0.00, max_run=8  (all tails)
]
# fmt: on


def main() -> None:
    pairs = [
        {"sequence_a": a, "sequence_b": b}
        for a, b in itertools.permutations(SEQUENCES, 2)
    ]
    out_path = Path(__file__).parent / "candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Wrote {len(pairs)} candidate pairs to {out_path}")


if __name__ == "__main__":
    main()
