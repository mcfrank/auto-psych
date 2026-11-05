# Inner Model Loop Report

Each model below is ONE distinct cognitive hypothesis. The posterior mass shows which single hypothesis best explains the data — it is **not** a recipe to combine the top models into a blend.

- Best model: **span_fallback_rule_violation** (posterior=1.000, elpd_loo=-4058.99)
- Trials: 7560
- Models compared: 3

## Posterior over models (ELPD-LOO)

| model | posterior | elpd_loo |
| --- | --- | --- |
| span_fallback_rule_violation | 1.0000 | -4058.99 |
| set_kin_consensus_violation_route | 0.0000 | -4075.31 |
| five_family_multiple_route | 0.0000 | -4077.06 |

## Hypotheses

- **span_fallback_rule_violation**: Refinement of the incumbent `coincidence_rule_violation_route`. People still judge a target by its partly averaged similarity to the examples, with distance-free arithmetic and round-number routes. Their decision is still more all-or-none after suspicious coincidences, and a target that breaks the set's most specific shared rule (same final digit, else all multiples of 5, else same parity) still loses a fixed share of its support. The one change adds a concept to that list. When two or more examples share no such rule, people fall back on the concept "numbers in the range the examples span", and a target outside that range loses the same fixed share of support, however near it is. So after incoherent sets such as "24, 35", "59, 14" or "48, 63, 53", targets below or above the examples (2, 4, 7, 96, 16) are rejected flatly, while targets between them stay graded by similarity. Single examples and coherent sets are unchanged.
- **set_kin_consensus_violation_route**: Refinement of the incumbent `coincidence_rule_violation_route` (partly averaged similarity to the examples on magnitude, final digit, parity and the set's common divisor, plus distance-free factor-kin, whole-set round and parity-preserving multiples routes; a suspicious coincidence makes the decision more decisive, and a target that breaks a multi-example set's coincidence rule loses a fixed share of its support). The one change, grafted from `set_kin_consensus_exemplar`: people credit an arithmetic kinship only to the extent that the target is kin to the set as a whole, so every distance-free route is weakened by the share of examples the target has no multiple, shared-factor or round-number relation to. Lone examples are judged as before, but after incoherent mixed sets a factor-kin or multiple of just one example (25 or 55 after "10, 3", 72 or 96 after "24, 35", 16 or 32 after "48, 63, 53", 7 after "59, 14") is rejected almost all-or-none, which is the over-graded incoherent-set polarization the critique reports.
- **five_family_multiple_route**: Refinement of the incumbent `coincidence_rule_violation_route` (partly averaged similarity to the examples on magnitude, final digit, parity and the set's common divisor, plus distance-free factor-kin, whole-set round and multiples routes, a fixed penalty for breaking the set's coincidence rule, and a decision that grows more decisive after suspicious coincidences). The one change concerns when a multiple feels like it has lost its example's character. People see a number ending in 5 as a "five-times-table" number rather than an odd one, and every multiple of it stays in that family. So the multiples route is cut for a parity switch only when the example is odd and does not end in 5: 34 after "17" and 86 after "43" are still rejected, but 20, 70 or 100 after "5", 50 after "25" and 90 after "15" keep the full route and are accepted almost all-or-none.

## Distinguishability (arviz.compare, PSIS-LOO)

`elpd_diff` and `dse` are relative to the best model. A model is only clearly worse than the best when `elpd_diff > 2 * dse`; models within ~2·dse of the top are statistically indistinguishable. `LOO reliable` is False when PSIS-LOO flagged this model's estimate as untrustworthy (many high Pareto-k points) — its row should be read with caution.

| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |
| --- | --- | --- | --- | --- | --- |
| span_fallback_rule_violation ←selected | 0.00 | 0.00 | — (best) | 1.000 | yes |
| set_kin_consensus_violation_route | 16.33 | 9.44 | no (within ~2·dse) | 0.000 | yes |
| five_family_multiple_route | 18.08 | 10.28 | no (within ~2·dse) | 0.000 | yes |
