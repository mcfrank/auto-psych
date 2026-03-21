# Design Rationale — Experiment 2

## Stimulus selection

**30 trials** selected from ~56,000 candidates (all same-length and mixed-length pairs of sequences with lengths 4, 6, 8) using Expected Information Gain (EIG) under a uniform prior over five cognitive models.

## EIG results

All 30 selected stimuli achieve the **maximum attainable EIG of 0.2887 bits** under the five models
(`griffiths_representativeness`, `alternation_bias`, `balance_heuristic`, `griffiths_v2`,
`weighted_balance_alternation`).

## Structure of selected stimuli

Every selected trial is a **length-8 pair** in which:
- One sequence is maximally alternating and perfectly balanced: `HTHTHTHT` or `THTHTHTH`
  (alternation rate = 1.0, balance = 0.5)
- The other sequence is **also perfectly balanced** (4H/4T) but with lower alternation rate
  (18 distinct patterns with alternation rates ranging from 3/7 to 5/7)

This contrast structure is optimal because:
1. **Balance is held constant** — both sequences in each pair have equal H/T counts (4:4),
   so balance-only models (`balance_heuristic`) predict chance (p=0.5). This creates maximal
   spread across models: the alternation-sensitive models differ from the balance-only model.
2. **Alternation is maximally varied** — one sequence alternates every flip; the other varies.
   `alternation_bias` strongly prefers the alternating sequence; `griffiths_representativeness`
   penalizes the extreme alternation rate (deviation from 0.5); `griffiths_v2` and
   `weighted_balance_alternation` are intermediate.
3. **All five models make meaningfully different predictions** on these pairs, maximizing the
   information each response provides about which model is correct.

## Why length-8 pairs dominate

Shorter sequences (length 4 or 6) have fewer possible statistics, causing more model predictions
to coincide. Length-8 sequences allow fine-grained alternation-rate differences while keeping
both sequences balanced — the combination that best separates the five models.

## Experiment 1 context

Experiment 1 found that balance explains most variance (r = 0.63) while alternation showed no
predictive power (r = −0.056). Experiment 2 focuses on pairs where balance cannot distinguish
models, forcing participants to rely on (or ignore) alternation structure. This will directly
test whether experiment 1's balance dominance reflects a genuine balance-only heuristic or
whether alternation matters when balance is equated.
