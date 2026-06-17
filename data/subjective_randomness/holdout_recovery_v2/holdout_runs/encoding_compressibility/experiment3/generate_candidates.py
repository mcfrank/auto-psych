#!/usr/bin/env python3
"""Generate candidate stimulus pairs for encoding_compressibility experiment 3.

The 6 competing models and what they care about:
  - prototype_similarity:
      Non-monotonic alternation penalty (|p_alts - theta_alt|) + imbalance (linear).
      Prefers moderate alternation (~0.65) AND balance.
  - inner_loop_model:
      Same as prototype_similarity but uses squared deviations.
  - rle_description_length:
      Strictly monotonic: (alts + 1) / n. More alternations always = more random.
  - max_run_length:
      Only cares about max_run_norm. Shorter max run = more random.
  - bayesian_diagnosticity:
      Fair-coin diagnosticity over alternating/streaky/biased alternatives.
      Penalizes very high (>0.9) and very low (<0.1) alternation rates.
  - head_balance (NEW in experiment 3):
      Only cares about imbalance (head proportion close to 0.5).
      Completely ignores alternation rate and run structure.

Key discriminating contrasts targeted:
  1. head_balance vs. rle/max_run: same imbalance, different alternation rate / max run
     (head_balance neutral; rle and max_run strongly prefer one side)
  2. head_balance vs. prototype: same imbalance, moderate vs. high alternation
     (head_balance neutral; prototype prefers moderate near theta_alt, rle prefers high)
  3. Monotonic vs. non-monotonic alternation: high-alt vs. moderate-alt balanced seqs
     (rle prefers high; prototype may prefer moderate near theta_alt=0.65)
  4. Max-run vs. alternation-rate: same max_run but different alts, or vice versa
  5. Imbalance vs. alternation tradeoff: balanced+streaky vs. imbalanced+alternating
  6. Bayesian vs. prototype: sequences where fair-coin diagnosticity diverges from
     prototype-similarity scores
"""

import itertools
import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "design" / "candidates.json"


def compute_features(seq: str) -> dict:
    s = seq.strip().upper()
    n = len(s)
    h = sum(1 for c in s if c == "H")
    alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
    max_run = 1
    cur = 1
    for i in range(1, n):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return {
        "n": n,
        "h": h,
        "alts": alts,
        "max_run": max_run,
        "p_alts": alts / (n - 1) if n > 1 else 0.0,
        "max_run_norm": (max_run - 1) / (n - 1) if n > 1 else 0.0,
        "imbalance": 2.0 * abs(h / n - 0.5) if n > 0 else 0.0,
        "rle_score": (alts + 1) / n,
    }


def _proto_score(f: dict, theta_alt: float = 0.65, alt_w: float = 0.5) -> float:
    return -(
        (1 - alt_w) * f["imbalance"] + alt_w * abs(f["p_alts"] - theta_alt)
    )


def _bayes_score(f: dict) -> float:
    # Approximation: penalize deviation from p_alts=0.5 and imbalance
    return -(0.6 * abs(f["p_alts"] - 0.5) + 0.3 * f["imbalance"])


def _head_score(f: dict) -> float:
    # head_balance: only cares about imbalance, ignores alternation entirely
    return -f["imbalance"]


def model_preferences(fa: dict, fb: dict) -> dict:
    """Return signed preference for each model (positive = A preferred)."""
    return {
        "proto": _proto_score(fa) - _proto_score(fb),
        "maxrun": -(fa["max_run_norm"] - fb["max_run_norm"]),
        "rle": fa["rle_score"] - fb["rle_score"],
        "bayes": _bayes_score(fa) - _bayes_score(fb),
        "head": _head_score(fa) - _head_score(fb),
    }


def is_discriminating(prefs: dict, threshold: float = 0.06) -> bool:
    """True if at least one model prefers A and at least one prefers B."""
    vals = list(prefs.values())
    return any(v > threshold for v in vals) and any(v < -threshold for v in vals)


def discrimination_score(prefs: dict) -> float:
    """Higher = stronger disagreement between models."""
    vals = list(prefs.values())
    positives = [v for v in vals if v > 0]
    negatives = [v for v in vals if v < 0]
    if not positives or not negatives:
        return 0.0
    # Score by the weakest "winning" side — ensures both sides have non-trivial signal
    return min(max(positives), -min(negatives))


def main() -> None:
    # Enumerate all H/T sequences of length 4–8, drop all-H and all-T
    all_seqs = []
    for n in range(4, 9):
        for bits in itertools.product("HT", repeat=n):
            seq = "".join(bits)
            f = compute_features(seq)
            if f["h"] == 0 or f["h"] == f["n"]:
                continue
            all_seqs.append((seq, f))

    # Score every ordered pair (i < j) for model discrimination
    scored = []
    seen: set = set()
    for i, (sa, fa) in enumerate(all_seqs):
        for j, (sb, fb) in enumerate(all_seqs):
            if i >= j:
                continue
            key = (sa, sb)
            if key in seen:
                continue
            seen.add(key)
            prefs = model_preferences(fa, fb)
            if is_discriminating(prefs):
                scored.append((discrimination_score(prefs), sa, sb))

    scored.sort(reverse=True)

    # Take top 200
    top = scored[:200]

    candidates = [{"sequence_a": sa, "sequence_b": sb} for _, sa, sb in top]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(candidates, indent=2))
    print(f"Wrote {len(candidates)} candidate pairs → {OUTPUT_PATH}")

    # Print a summary of the top 10 pairs
    print(
        f"\nTop 10 most discriminating pairs:"
        f"\n{'Score':>8}  {'Seq A':>10}  {'Seq B':>10}"
        f"  {'Proto':>6}  {'MaxRun':>6}  {'RLE':>6}  {'Bayes':>6}  {'Head':>6}"
    )
    for score, sa, sb in scored[:10]:
        fa_feats = compute_features(sa)
        fb_feats = compute_features(sb)
        prefs = model_preferences(fa_feats, fb_feats)
        print(
            f"{score:>8.3f}  {sa:>10}  {sb:>10}"
            f"  {prefs['proto']:>+6.3f}  {prefs['maxrun']:>+6.3f}"
            f"  {prefs['rle']:>+6.3f}  {prefs['bayes']:>+6.3f}"
            f"  {prefs['head']:>+6.3f}"
        )


if __name__ == "__main__":
    main()
