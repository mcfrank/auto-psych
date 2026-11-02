---
name: exp3-iter1-process-probes
description: exp3 iter_1 cand_0 (2026-11-01) MAP probes on graded_prior_digit_sum_concepts (MAP -3513.54) — target-as-evidence, previous-set carry-over, random-generator program, odd own family, kind-level sharpening all pin at normative
metadata:
  type: project
---
Base graded_prior_digit_sum_concepts, MAP LL -3513.54 on exp3 (family weights: parity .025, special .152, multiples .125, ends .142, contains ~0, exceptions ~0, digit sums .025, ranges .53; scale 79, centre .017, dsum_decay -0.25).
- Target as partial evidence (positive test, boost (P(t|h)/P(t|noise))^w): w 0.006, no gain.
- Carry-over of the previous different set's family posterior (sequential priming): log-share power and linear blend both optimal at 0; even 0.02 blend costs 3 nats.
- Explicit "random generator" program with learned per-example typicality: weight 0, typicality 0.
- Odd and even as separate families: no gain (odd .013); removing odd costs 230 nats.
- Two-stage inference (sharpen across kinds, Bayes within): exponent 1.03, +0.4 nats.
- Incumbent aggregates are calibrated (inside/below-min/near-last/odd-set-even-target etc. all |z|<1.6).
- PROBE BUG to avoid: model.compile_fn(model["p_yes"]) draws random RVs; wrap with model.replace_rvs_by_values first.
**How to apply:** the exp3 normative concept learner is saturated on decision-, sequence- and target-side mechanisms too. See [[exp3-decision-rule-probes]], [[exp3-concept-family-probes]].
