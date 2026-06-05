# Design rationale — experiment 1, subjective_randomness

## Stimulus generation

Candidates were drawn from a hand-curated set of 24 coin-flip sequences spanning lengths 3, 4, 6, and 8, covering the full feature space of the three cognitive models:

| Feature | Model |
|---|---|
| `p_alts` (alternation proportion) | `alternation_bias` |
| `max_run` (longest streak) | `runs_penalty` |
| `h/n` (head proportion), `n` (length) | `bayesian_fair_coin` |

The 24 sequences included: perfectly alternating (p_alts = 1.0), all-H and all-T extremes, long-run-biased sequences (e.g., HTTTTTTH, max_run = 6), and intermediate sequences. All 552 ordered (A, B) pairs were scored by EIG.

## EIG scoring

EIG was computed via prior-predictive sampling (no MCMC fitting), using a uniform prior over the three models (the model registry has no fitted posteriors yet). EIG range across the top 20 selected stimuli: **0.1657 – 0.2061 bits**.

## Selected stimuli and discriminating logic

The top-EIG pair (EIG = 0.206) is **TTT vs HTTTTTTH**. This pair maximally discriminates because:

- **`alternation_bias`** prefers HTTTTTTH (p_alts = 0.286) over TTT (p_alts = 0.00).
- **`runs_penalty`** prefers TTT (max_run = 3) over HTTTTTTH (max_run = 6).
- **`bayesian_fair_coin`** contrasts two tail-biased sequences of very different length (n = 3 vs n = 8).

The second-tier pairs (EIG 0.165–0.176) contrast all-T sequences of different lengths (TTT vs TTTTTTTT) and all-H short vs all-T long (HHH vs TTTTTTTT). These discriminate primarily between `bayesian_fair_coin` (length-sensitive) and the alternation/runs models (length-insensitive when p_alts = 0 or max_run saturates).

The perfectly alternating sequences (HTHTHTHT, HTHTHT, HTHT) appear paired with TTTTTTTT in the top 20 — contrasting maximum p_alts with zero p_alts, again pitting alternation_bias against runs_penalty and bayesian_fair_coin.

## Summary

- **N stimuli selected**: 20 (within the 30-trial budget).
- **EIG range**: 0.1657 – 0.2061 bits (spread = 0.040 bits).
- **Discriminating design axis**: all top pairs exploit disagreement between the alternation dimension (p_alts), the streak dimension (max_run), and the Bayesian likelihood dimension (h/n, n).
- **Position symmetry**: both (A, B) and (B, A) orderings of high-EIG pairs appear in the set, ensuring position-balanced coverage.
