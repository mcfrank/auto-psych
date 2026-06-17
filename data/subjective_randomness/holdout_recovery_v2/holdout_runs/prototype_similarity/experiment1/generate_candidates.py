"""Generate candidate stimulus pairs for EIG scoring.

Discriminating target: encoding_compressibility vs bayesian_diagnosticity.

Key divergence axis:
- EC penalizes runs + periodicity + imbalance
- BD penalizes alternating/streaky/biased patterns via likelihood ratios

Alternating sequences (low max_run, balanced) are mildly penalized by EC but
strongly penalized by BD. Pairing them against balanced moderate sequences
should produce divergent prior-predictive predictions.

Categories generated:
1. Highly alternating vs balanced-moderate (core discriminator)
2. Periodic (period > 2) vs balanced-random-looking
3. Streaky vs balanced (both models agree, but included for coverage)
4. Imbalanced vs balanced (both penalize imbalance differently)
5. Cross-length pairs exploiting length-normalization differences
"""

import itertools
import json
import os


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
        "imbalance": 2.0 * abs(h / n - 0.5) if n > 0 else 0.0,
        "periodicity": _periodicity(s),
    }


def _enumerate_seqs(min_len=4, max_len=8):
    seqs = []
    for length in range(min_len, max_len + 1):
        for bits in itertools.product("HT", repeat=length):
            seqs.append("".join(bits))
    return seqs


def main():
    seqs = _enumerate_seqs(4, 8)
    by_seq = {s: _features(s) for s in seqs}

    # ---- classify by archetype ----------------------------------------

    # Highly alternating: p_alts > 0.85, imbalance < 0.15 (low max_run by definition)
    highly_alt = [s for s, f in by_seq.items() if f["p_alts"] > 0.85 and f["imbalance"] < 0.15]

    # Balanced moderate: 0.3 <= p_alts <= 0.65, imbalance < 0.15, max_run <= 3
    bal_mod = [
        s for s, f in by_seq.items()
        if 0.3 <= f["p_alts"] <= 0.65 and f["imbalance"] < 0.15 and f["max_run"] <= 3
    ]

    # Periodic non-alternating: periodicity > 0.7, 0.2 < p_alts < 0.75, imbalance < 0.15
    periodic_nonalt = [
        s for s, f in by_seq.items()
        if f["periodicity"] > 0.7 and 0.2 < f["p_alts"] < 0.75 and f["imbalance"] < 0.15
    ]

    # Balanced low-alternation: 0.1 <= p_alts <= 0.35, imbalance < 0.15
    bal_low_alt = [
        s for s, f in by_seq.items()
        if 0.1 <= f["p_alts"] <= 0.35 and f["imbalance"] < 0.15
    ]

    # Streaky: p_alts < 0.15 (very low alternation)
    streaky = [s for s, f in by_seq.items() if f["p_alts"] < 0.15]

    # Imbalanced balanced: imbalance > 0.3, length 4-6
    imbalanced = [s for s, f in by_seq.items() if f["imbalance"] > 0.3 and f["n"] <= 6]

    # Balanced random-looking: p_alts in [0.4, 0.65], imbalance < 0.1, periodicity < 0.5
    random_looking = [
        s for s, f in by_seq.items()
        if 0.4 <= f["p_alts"] <= 0.65 and f["imbalance"] < 0.1 and f["periodicity"] < 0.5
    ]

    pairs = []
    seen = set()

    def add_pair(a, b):
        if a != b:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                pairs.append({"sequence_a": a, "sequence_b": b})

    # ---- Category 1: Highly alternating vs balanced-moderate (core) ----
    # BD hates alternating; EC mildly penalizes (periodicity) but likes low max_run
    import random
    rng = random.Random(42)

    alt_sample = rng.sample(highly_alt, min(20, len(highly_alt)))
    bm_sample = rng.sample(bal_mod, min(30, len(bal_mod)))
    for a in alt_sample:
        for b in bm_sample[:8]:
            add_pair(a, b)

    # ---- Category 2: Highly alternating vs random-looking --------------
    rl_sample = rng.sample(random_looking, min(20, len(random_looking)))
    for a in alt_sample[:12]:
        for b in rl_sample[:6]:
            add_pair(a, b)

    # ---- Category 3: Periodic (non-alternating) vs random-looking ------
    # EC penalizes periodicity; BD penalizes based on alts count
    per_sample = rng.sample(periodic_nonalt, min(20, len(periodic_nonalt)))
    for a in per_sample:
        for b in rl_sample[:5]:
            add_pair(a, b)

    # ---- Category 4: Periodic non-alternating vs balanced-moderate -----
    for a in per_sample[:12]:
        for b in bm_sample[:5]:
            add_pair(a, b)

    # ---- Category 5: Highly alternating vs balanced-low-alt ------------
    # BD: alternating vs non-alternating; EC: both have low max_run (similar penalty)
    bla_sample = rng.sample(bal_low_alt, min(15, len(bal_low_alt)))
    for a in alt_sample[:10]:
        for b in bla_sample[:5]:
            add_pair(a, b)

    # ---- Category 6: Streaky vs balanced-moderate ----------------------
    str_sample = rng.sample(streaky, min(10, len(streaky)))
    for a in str_sample:
        for b in bm_sample[:5]:
            add_pair(a, b)

    # ---- Category 7: Imbalanced vs balanced alternating ----------------
    # EC penalizes imbalance; BD penalizes biased
    imb_sample = rng.sample(imbalanced, min(10, len(imbalanced)))
    for a in imb_sample:
        for b in alt_sample[:5]:
            add_pair(a, b)

    # ---- Category 8: Cross-length pairs --------------------------------
    # Length 4 alternating vs length 7-8 balanced-moderate
    short_alt = [s for s in highly_alt if len(s) <= 5]
    long_bm = [s for s in bal_mod if len(s) >= 7]
    for a in short_alt[:8]:
        for b in rng.sample(long_bm, min(5, len(long_bm))):
            add_pair(a, b)

    # Length 7-8 alternating vs short balanced
    long_alt = [s for s in highly_alt if len(s) >= 7]
    short_bm = [s for s in bal_mod if len(s) <= 5]
    for a in long_alt[:8]:
        for b in rng.sample(short_bm, min(5, len(short_bm))):
            add_pair(a, b)

    # ---- Cap at 250 -------------------------------------------------------
    rng.shuffle(pairs)
    pairs = pairs[:250]

    out_path = os.path.join(EXP_DIR, "design", "candidates.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} candidates to {out_path}")

    # Print summary stats
    for cat, seqs_list, label in [
        (highly_alt, alt_sample, "highly_alternating"),
        (bal_mod, bm_sample, "balanced_moderate"),
        (periodic_nonalt, per_sample, "periodic_nonalt"),
        (random_looking, rl_sample, "random_looking"),
        (streaky, str_sample, "streaky"),
    ]:
        print(f"  {label}: {len(seqs_list)} archetypes ({len(cat)} total in class)")


if __name__ == "__main__":
    main()
