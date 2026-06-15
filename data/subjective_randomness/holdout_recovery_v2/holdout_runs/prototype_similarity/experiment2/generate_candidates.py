"""Generate candidate stimulus pairs for EIG scoring.

4 competing models:
  - alternation_prototype: p_left ~ |p_alts_a - theta_alt| L1 distance (theta_alt ~ Uniform(0.35, 0.95))
  - inner_loop_model: learned Bayesian diagnosticity (best from exp1)
  - bayesian_diagnosticity: fixed-param Bayesian diagnosticity
  - encoding_compressibility: penalizes runs + periodicity + imbalance

Key discriminating pair categories:
  1. Medium-alternating vs fair-balanced (CORE: AP vs Bayesian)
       AP prefers medium-alt (p_alts ~0.6-0.8, close to theta_alt)
       Bayesian prefers fair (p_alts ~0.4-0.55)
  2. Periodic-medium-alt vs aperiodic-medium-alt (AP vs EC)
       AP: indifferent (same p_alts)
       EC: prefers aperiodic (lower periodicity penalty)
  3. Highly alternating vs medium-alternating (AP preference shape)
       AP: prefers medium (theta_alt ~0.7 means |1.0-0.7| > |0.7-0.7|)
       Bayesian: dislikes both, but especially highly-alt
  4. Highly alternating vs balanced-moderate (BD/inner vs EC — from exp1)
  5. Medium-alternating vs streaky (broad coverage)
  6. Cross-length medium-alt vs fair
"""

import itertools
import json
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

    # Medium-high alternating: p_alts in [0.55, 0.85], balanced
    # AP ideal range: close to expected theta_alt ~0.65-0.75
    medium_high_alt = [
        s for s, f in by_seq.items()
        if 0.55 <= f["p_alts"] <= 0.85 and f["imbalance"] < 0.15
    ]

    # Fair-balanced: p_alts in [0.35, 0.55], balanced, no long runs
    # Bayesian ideal: close to fair-coin (p_alts ~ 0.5)
    fair_balanced = [
        s for s, f in by_seq.items()
        if 0.35 <= f["p_alts"] <= 0.55 and f["imbalance"] < 0.1 and f["max_run"] <= 3
    ]

    # Highly alternating: p_alts > 0.85, balanced
    # Too alternating even for AP (if theta_alt ~ 0.7, |1.0-0.7| > |0.7-0.7|)
    highly_alt = [
        s for s, f in by_seq.items()
        if f["p_alts"] > 0.85 and f["imbalance"] < 0.15
    ]

    # Periodic-medium-alt: same p_alts range as medium_high_alt but periodic
    # AP: similar score to aperiodic peers; EC: penalizes periodicity
    periodic_mid_alt = [
        s for s, f in by_seq.items()
        if 0.50 <= f["p_alts"] <= 0.85 and f["periodicity"] > 0.6 and f["imbalance"] < 0.15
    ]

    # Aperiodic-medium-alt: same p_alts range but NOT periodic
    # AP: similar p_alts as periodic peers; EC: lower penalty
    aperiodic_mid_alt = [
        s for s, f in by_seq.items()
        if 0.55 <= f["p_alts"] <= 0.85 and f["periodicity"] < 0.25 and f["imbalance"] < 0.15
    ]

    # Balanced-moderate (from exp1 — BD/inner vs EC axis)
    bal_mod = [
        s for s, f in by_seq.items()
        if 0.30 <= f["p_alts"] <= 0.65 and f["imbalance"] < 0.15 and f["max_run"] <= 3
    ]

    # Streaky: very low alternation
    streaky = [s for s, f in by_seq.items() if f["p_alts"] < 0.20]

    pairs = []
    seen = set()

    def add_pair(a, b):
        if a != b:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                pairs.append({"sequence_a": a, "sequence_b": b})

    rng = random.Random(42)

    mha_sample = rng.sample(medium_high_alt, min(25, len(medium_high_alt)))
    fb_sample = rng.sample(fair_balanced, min(30, len(fair_balanced)))
    ha_sample = rng.sample(highly_alt, min(20, len(highly_alt)))
    pma_sample = rng.sample(periodic_mid_alt, min(20, len(periodic_mid_alt)))
    apma_sample = rng.sample(aperiodic_mid_alt, min(20, len(aperiodic_mid_alt)))
    bm_sample = rng.sample(bal_mod, min(25, len(bal_mod)))
    str_sample = rng.sample(streaky, min(10, len(streaky)))

    # ---- Category 1: Medium-high alternating vs fair-balanced (CORE) ---
    # AP likes medium-alt; Bayesian models like fair-balanced
    for a in mha_sample:
        for b in fb_sample[:10]:
            add_pair(a, b)

    # ---- Category 2: Periodic-medium-alt vs aperiodic-medium-alt -------
    # AP: similar predictions (same p_alts range)
    # EC: strongly prefers aperiodic (lower periodicity penalty)
    for a in pma_sample:
        for b in apma_sample[:8]:
            add_pair(a, b)

    # ---- Category 3: Highly alternating vs medium-alternating ----------
    # AP: prefers medium (theta_alt ~0.7, so |0.7-0.7|=0 < |1.0-0.7|=0.3)
    # Bayesian: dislikes both but especially highly-alt
    for a in ha_sample:
        for b in mha_sample[:8]:
            add_pair(a, b)

    # ---- Category 4: Highly alternating vs balanced-moderate -----------
    # BD/inner dislikes highly-alt; EC only mildly penalizes (low runs + periodicity)
    for a in ha_sample[:15]:
        for b in bm_sample[:6]:
            add_pair(a, b)

    # ---- Category 5: Medium-high alternating vs streaky ----------------
    # Both AP and Bayesian prefer medium-alt; EC also penalizes streaky (long runs)
    # All models agree — included for coverage calibration
    for a in mha_sample[:10]:
        for b in str_sample:
            add_pair(a, b)

    # ---- Category 6: Cross-length medium-alt vs fair-balanced ----------
    short_mha = [s for s in medium_high_alt if len(s) <= 5]
    long_fb = [s for s in fair_balanced if len(s) >= 7]
    for a in rng.sample(short_mha, min(8, len(short_mha))):
        for b in rng.sample(long_fb, min(5, len(long_fb))):
            add_pair(a, b)

    long_mha = [s for s in medium_high_alt if len(s) >= 7]
    short_fb = [s for s in fair_balanced if len(s) <= 5]
    for a in rng.sample(long_mha, min(8, len(long_mha))):
        for b in rng.sample(short_fb, min(5, len(short_fb))):
            add_pair(a, b)

    # ---- Category 7: Fair-balanced vs streaky --------------------------
    # All models prefer fair-balanced; included for calibration
    for a in fb_sample[:8]:
        for b in str_sample[:5]:
            add_pair(a, b)

    # ---- Cap at 250 ----------------------------------------------------
    rng.shuffle(pairs)
    pairs = pairs[:250]

    out_path = os.path.join(EXP_DIR, "design", "candidates.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} candidates to {out_path}")
    print(f"  medium_high_alt: {len(mha_sample)} sampled ({len(medium_high_alt)} total)")
    print(f"  fair_balanced: {len(fb_sample)} sampled ({len(fair_balanced)} total)")
    print(f"  highly_alt: {len(ha_sample)} sampled ({len(highly_alt)} total)")
    print(f"  periodic_mid_alt: {len(pma_sample)} sampled ({len(periodic_mid_alt)} total)")
    print(f"  aperiodic_mid_alt: {len(apma_sample)} sampled ({len(aperiodic_mid_alt)} total)")
    print(f"  bal_mod: {len(bm_sample)} sampled ({len(bal_mod)} total)")
    print(f"  streaky: {len(str_sample)} sampled ({len(streaky)} total)")


if __name__ == "__main__":
    main()
