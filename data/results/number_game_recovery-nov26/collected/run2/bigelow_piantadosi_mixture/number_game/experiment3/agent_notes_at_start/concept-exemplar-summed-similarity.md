---
name: concept-exemplar-summed-similarity
description: Exemplar model with concept-mediated (Tenenbaum single-example) similarity, averaged over examples, fits ~-1141 LL — far behind Bayesian intersection (~-908)
metadata:
  type: project
---
Round 3 cand_1 (2026-10-30): `concept_mediated_exemplar_similarity` — similarity(x,y) = P(y in concept | x alone) over 45 rules + intervals, averaged over examples, Luce gain/criterion. MAP LL -1141.5 (3 params), smoke fit 2 s. Beats feature exemplar models (-1213/-1232) but ~230 nats behind the Bayesian set-level models.

**Why:** summing similarity across examples gives up the requirement that one concept covers all examples; the data need that.
**How to apply:** additive exemplar accounts are ruled out; don't retry summed-similarity variants. Related: [[all-or-none-rule-commitment-fails]], [[digit-concepts-key]].
