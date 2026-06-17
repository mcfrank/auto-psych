"""Generate candidate stimulus pairs for bayesian_diagnosticity experiment 2.

Five competing models are active:
  prototype_similarity     — imbalance + p_alts (linear absolute deviation)
  encoding_compressibility — max_run_norm + periodicity + imbalance
  inner_loop_model         — p_alts only (quadratic deviation)
  length_sensitive_alt     — count-scale alternation (alts, n)
  bayesian_markov_fairness — log-Bayes-factor (alts, n)

Four discrimination strategies drive candidate selection:

  S1  Cross-length: LSA (count scale) vs ILM (proportion scale)
      Pairs where M4 prefers A but M3 prefers B (or vice versa) because
      the count deviation and proportion deviation rank sequences differently
      when sequences have different lengths.

  S2  Imbalance: prototype_similarity vs inner_loop_model
      Same-length pairs with large imbalance difference but similar p_alts.
      PS penalises imbalance; ILM ignores it.

  S3  Structural: encoding_compressibility vs proportion-based models
      Pairs differing strongly in periodicity or max_run_norm while holding
      p_alts similar. Encoding penalises structure; others ignore it.

  S4  BMF vs LSA on same-length sequences
      Pairs where one sequence sits near the fair-coin ideal (alts≈(n-1)/2)
      and the other sits near the LSA ideal (alts≈0.65*(n-1)).
"""
from __future__ import annotations
import json
import random
from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path("/Users/ben/Documents/auto-psych")
sys.path.insert(0, str(REPO_ROOT))
from src.pipelines.outer_loop.projects.subjective_randomness.preprocess import featurize_stimulus  # noqa: E402

EXP_DIR = Path(__file__).parent
THETA_ALT = 0.65   # prior centre for alternation ideal
MAX_PAIRS = 250
RANDOM_SEED = 42


def _seq_feats(seq: str) -> dict:
    f = featurize_stimulus(seq, "HHHH")
    return {
        "n": f["n_a"],
        "alts": f["alts_a"],
        "p_alts": f["p_alts_a"],
        "imbalance": f["imbalance_a"],
        "max_run_norm": f["max_run_norm_a"],
        "periodicity": f["periodicity_a"],
    }


def _profile(seq: str) -> tuple:
    f = _seq_feats(seq)
    return (
        f["n"],
        round(f["p_alts"], 6),
        round(f["imbalance"], 6),
        round(f["max_run_norm"], 6),
        round(f["periodicity"], 6),
    )


def _canonical_seqs() -> list[str]:
    """One representative sequence per distinct feature profile, lengths 4–8."""
    seqs, seen = [], set()
    for n in range(4, 9):
        for bits in product("HT", repeat=n):
            seq = "".join(bits)
            key = _profile(seq)
            if key not in seen:
                seen.add(key)
                seqs.append(seq)
    return seqs


def _build_pairs(seqs: list[str]) -> list[dict]:
    random.seed(RANDOM_SEED)
    feats = {s: _seq_feats(s) for s in seqs}
    pool: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        pool.add((min(a, b), max(a, b)))

    # S1 — Cross-length: LSA vs ILM disagree
    s1: list[tuple[str, str]] = []
    for a in seqs:
        fa = feats[a]
        for b in seqs:
            if a == b:
                continue
            fb = feats[b]
            if abs(fa["n"] - fb["n"]) < 2:
                continue
            cd_a = (fa["alts"] - THETA_ALT * (fa["n"] - 1)) ** 2
            cd_b = (fb["alts"] - THETA_ALT * (fb["n"] - 1)) ** 2
            pd_a = (fa["p_alts"] - THETA_ALT) ** 2
            pd_b = (fb["p_alts"] - THETA_ALT) ** 2
            # Models disagree on which sequence is better
            if (cd_a < cd_b) != (pd_a < pd_b):
                # Require non-trivial signal in both dimensions
                if abs(cd_a - cd_b) > 0.4 and abs(pd_a - pd_b) > 0.003:
                    s1.append((min(a, b), max(a, b)))
    s1 = list(dict.fromkeys(s1))  # deduplicate while preserving order
    random.shuffle(s1)
    for pair in s1[:100]:
        pool.add(pair)

    # S2 — Imbalance: prototype_similarity vs inner_loop_model
    s2: list[tuple[str, str]] = []
    for i, a in enumerate(seqs):
        fa = feats[a]
        for b in seqs[i + 1:]:
            fb = feats[b]
            if abs(fa["imbalance"] - fb["imbalance"]) > 0.3 and abs(fa["p_alts"] - fb["p_alts"]) < 0.2:
                s2.append((min(a, b), max(a, b)))
    random.shuffle(s2)
    for pair in s2[:70]:
        pool.add(pair)

    # S3 — Structural: encoding_compressibility vs proportion models
    s3: list[tuple[str, str]] = []
    for i, a in enumerate(seqs):
        fa = feats[a]
        for b in seqs[i + 1:]:
            fb = feats[b]
            period_diff = abs(fa["periodicity"] - fb["periodicity"])
            run_diff = abs(fa["max_run_norm"] - fb["max_run_norm"])
            p_alts_diff = abs(fa["p_alts"] - fb["p_alts"])
            if (period_diff > 0.4 or run_diff > 0.4) and p_alts_diff < 0.25:
                s3.append((min(a, b), max(a, b)))
    random.shuffle(s3)
    for pair in s3[:80]:
        pool.add(pair)

    # S4 — BMF vs LSA: fair-coin ideal vs alternation ideal, same length
    for i, a in enumerate(seqs):
        fa = feats[a]
        for b in seqs[i + 1:]:
            fb = feats[b]
            if fa["n"] != fb["n"]:
                continue
            n = fa["n"]
            fair_ideal = (n - 1) / 2.0
            lsa_ideal = THETA_ALT * (n - 1)
            a_near_fair = abs(fa["alts"] - fair_ideal) < 1.0
            b_near_lsa = abs(fb["alts"] - lsa_ideal) < 1.5
            b_near_fair = abs(fb["alts"] - fair_ideal) < 1.0
            a_near_lsa = abs(fa["alts"] - lsa_ideal) < 1.5
            if (a_near_fair and b_near_lsa) or (b_near_fair and a_near_lsa):
                pool.add((min(a, b), max(a, b)))

    pairs = [{"sequence_a": a, "sequence_b": b} for a, b in sorted(pool)]
    if len(pairs) > MAX_PAIRS:
        random.shuffle(pairs)
        pairs = pairs[:MAX_PAIRS]
    return pairs


def main() -> None:
    seqs = _canonical_seqs()
    pairs = _build_pairs(seqs)

    out = EXP_DIR / "design" / "candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(pairs)} candidate pairs → {out}")
    # Strategy breakdown summary
    print(f"Canonical sequences across lengths 4-8: {len(seqs)}")


if __name__ == "__main__":
    main()
