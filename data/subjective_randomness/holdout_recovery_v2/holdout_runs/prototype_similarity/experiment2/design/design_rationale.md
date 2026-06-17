# Design rationale — experiment 2 (prototype_similarity)

## Setup

250 candidate pairs generated; 20 selected by EIG.

**4 competing models** (uniform prior — no experiment 1 posterior available in registry):
- `alternation_prototype` (new): L1 distance from learned theta_alt ~ Uniform(0.35, 0.95)
- `inner_loop_model`: learned Bayesian diagnosticity (best model from exp1, 59.3% posterior)
- `bayesian_diagnosticity`: fixed-parameter Bayesian diagnosticity
- `encoding_compressibility`: penalizes runs, periodicity, imbalance

## EIG range

**0.0926 – 0.1795** (top 20 stimuli); higher ceiling than experiment 1 (0.0945–0.1370), reflecting the additional discriminability introduced by the `alternation_prototype` model.

## Selected stimulus patterns

All 20 top-EIG stimuli pair a **medium-alternating sequence** (p_alts ~0.57–0.86) against a **streaky or highly-imbalanced sequence** (p_alts ~0.0–0.14 or heavy imbalance):

| sequence_a (medium-alt) | sequence_b (streaky/imbalanced) | EIG |
|---|---|---|
| HHTHTHT | HTTTTTTT | 0.180 |
| THTHHTH | HTTTTTTT | 0.169 |
| THHTHTH | TTTT | 0.166 |
| HTHHTHT | HHHHH | 0.158 |
| … | … | … |

## Why these pairs are diagnostic

The models diverge in **how confidently** they prefer the medium-alternating sequence over the streaky one:

- **alternation_prototype**: medium-alt (p_alts ~0.7) falls close to the prior-typical theta_alt; streaky (p_alts ~0.0) is far away. Predicts a strong, graded preference.
- **inner_loop_model / bayesian_diagnosticity**: both sequences are non-random by Bayesian-diagnosticity reasoning, but they differ in how they compare to fair-coin vs alternating/streaky generators. The models assign different log-Bayes-factor rankings.
- **encoding_compressibility**: streaky sequences have large max_run and imbalance — heavily penalized. Medium-alt sequences have short runs — preferred. EC agrees on direction but with a different confidence profile driven by the run-length penalty.

The EIG is highest where `alternation_prototype` and the Bayesian models assign divergent p_left confidences — the medium-alt vs streaky contrast sits at that sweet spot.

## Candidate pool structure

| Archetype | N in class | Sampled |
|---|---|---|
| medium_high_alt (0.55 ≤ p_alts ≤ 0.85, balanced) | 80 | 25 |
| fair_balanced (0.35 ≤ p_alts ≤ 0.55, low max_run) | 22 | 22 |
| highly_alt (p_alts > 0.85, balanced) | 14 | 14 |
| periodic_mid_alt (p_alts 0.5–0.85, periodicity > 0.6) | 50 | 20 |
| aperiodic_mid_alt (p_alts 0.55–0.85, periodicity < 0.25) | 4 | 4 |
| bal_mod (p_alts 0.3–0.65, balanced, max_run ≤ 3) | 80 | 25 |
| streaky (p_alts < 0.20) | 36 | 10 |

Note: `aperiodic_mid_alt` class has only 4 members due to tight joint constraints (moderate p_alts AND low periodicity are naturally correlated with different length combinations). The Category 2 AP-vs-EC pairs are therefore sparse; this limitation is acceptable given the strong discrimination available from the AP-vs-Bayesian axis.
