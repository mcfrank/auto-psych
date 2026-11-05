---
name: common-divisor-kinship-exemplar
description: exp2 iter_1 c5 — set-gcd kinship (gcd(t,g)>=3, single example g=example) on common_divisor_rule base, MLE NLL 2707.8 vs incumbent 2714.3
metadata:
  type: project
---
Experiment 2, iter_1 candidate_5 (2026-11-02), `common_divisor_kinship_exemplar`, a refinement of `common_divisor_rule_exemplar`. The factor penalty applies when g = gcd(set) >= 3 (for a single example, g is the example) and gcd(t, g) < 3. In-sample MLE NLL was 2707.8, against 2714.3 for the incumbent common_divisor_sharpened_exemplar and 2720.9 for the rule model, with the same 9 parameters. The smoke fit was -2711.9.
Screened variants:
- incumbent + single-example gcd<3 only: 2712.5
- single example "multiples or factors of the example": 2717.8
- rule model with single-example gcd<3: no change, because no single-example target has gcd exactly 2
Most of the gain comes from the multi-example sets 28,98 / 84,56 / 90,45 / 80,70,90,100.
**Why:** aimed at the single_example_shared_factor_contrast critique (q=0.008) and the 90,45 problem noted in [[common-divisor-sharpened-exemplar]].
**How to apply:** see [[exp2-iter1-critique-stats]] and [[common-divisor-rule-exemplar]]. The screen script is in iter_1/candidate_5/scratch/screen.py.
