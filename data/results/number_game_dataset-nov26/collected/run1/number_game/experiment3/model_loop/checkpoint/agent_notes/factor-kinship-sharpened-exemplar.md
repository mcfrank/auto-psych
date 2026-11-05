---
name: factor-kinship-sharpened-exemplar
description: exp2 iter1 c4 — divisor penalty as gcd(t, set gcd)<3, singles included, on common_divisor_sharpened incumbent; +6.5 nats MAP
metadata:
  type: project
---
Experiment 2, iter_1 candidate_4 (2026-11-02) refined `common_divisor_sharpened_exemplar`. The penalty now fires when gcd(target, set gcd) < 3 with set gcd ≥ 3, and a single example counts as its own gcd. MAP results with no new parameters: incumbent -2714.4. Singles-only change (B) -2712.6. Multi-set kinship only (D) -2709.6. Both together (C, submitted) -2707.9. Smoke fit -2711.9.
**Why:** this targets the critique stat single_example_shared_factor_contrast and the "90, 45" failure noted in [[common-divisor-sharpened-exemplar]]. The single-example data is thin (34 stimuli), so most of the gain comes from multi-example sets.
**How to apply:** a separate strength for single-example sets has not been tried yet. See [[exp2-iter1-critique-stats]].
