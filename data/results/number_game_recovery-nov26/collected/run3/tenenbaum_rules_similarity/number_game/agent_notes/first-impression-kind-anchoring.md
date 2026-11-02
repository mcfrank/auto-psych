---
name: first-impression-kind-anchoring
description: exp3 iter_3 c0: kind log-posterior = blend of first-example-only and all-example kind posteriors (anchoring ~Beta(1.5,3)); smoke OK ll −3653.7
metadata:
  type: project
---
exp3 iter_3/candidate_0 (2026-10-31), first_impression_kind_anchoring. Built on rule_family_doubt_criterion. Kind log posterior (8 families + range) = a·log P(kind|x1) + (1−a)·log P(kind|all), renormalised. Within-kind concept weights come from all examples. prepare_observed adds the first example's per-rule log lik and its per-size interval evidence. Smoke check OK in about 35 s per run, ll −3653.7 (incumbent about −3652.8). The anchoring posterior and novelty were not checked.
Exp3 has only one permuted set pair (8,16 / 16,8), so any order effect here is confounded with the stimuli.
**How to apply:** if a→0 it duplicates the incumbent. See [[serial-position-reliability]], [[arbitrary-opening-example]].
