---
name: power-chain-generation
description: exp3 iter_2 c0 — powers-of-common-base chain concept, not overshadowed, on rule_ambiguity incumbent: +1.6 MAP, smoke -4673.6; other probes ~0 or worse
metadata:
  type: project
---
exp3 iter_2 candidate_0 (2026-11-02): this candidate adds a "chain" concept to rule_ambiguity_single_doubt. If all examples are powers of one base (smallest base b<=10), the other powers of b get their own salience, and a shared final digit does not overshadow them. Results: MAP +1.6, smoke -4673.6 vs -4675.4 for the incumbent. Signature: 64,4->16 observed .78 vs predicted .21; 81,9->27 observed 1.0 vs .53. Only about 200 rows qualify, so the gain is capped.

Other MAP probes on the same incumbent (MAP -4664.6), all ~0 or worse:
- conjunctive sampling link (lo+(hi-lo)*share^K): -0.4
- extra weight on the nearest example's own properties: +1.2
- dropping the magnitude band for n=1 (all or partial): about -1 to 0
- unnormalised for/against tally instead of share: -20

**Why:** the incumbent is near its fit ceiling on exp3, and most n=1 set-mean residuals look like binomial noise (9 reps per pair).
**How to apply:** the share normalisation is right; don't retry tally links or n=1 band removal. Related: [[rule-ambiguity-single-doubt]] [[exp3-n1-data-facts]]

Repair (2026-11-02, candidate_0_repair_1): the first version gave the chain its own Dirichlet salience component, and admission rejected it (R-hat 1.051, ESS 66). The fix was to take the chain out of the Dirichlet and set its weight to chain_boost * salience[powers], with chain_boost ~ LogNormal(0, .75). A 4x500 fit then gave R-hat <= 1.01, 0 divergences, boost about .9 ± .6. Smoke fit -4674.0.
**How to apply:** a concept that only a few sets qualify for should not get its own Dirichlet salience component. Tie it to an existing family with a log-normal multiplier.
