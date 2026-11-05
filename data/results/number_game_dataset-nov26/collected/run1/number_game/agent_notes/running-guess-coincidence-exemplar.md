---
name: running-guess-coincidence-exemplar
description: exp2 iter0 c2 — order-dependent coincidence: sharpening from the trailing run of examples sharing a property; smoke -2751.4
metadata:
  type: project
---
exp2 iter_0 candidate_2 (2026-11-02), `running_guess_coincidence_exemplar`: the same model as coincidence_sharpened_exemplar, except that the coincidence score is computed only over the longest run of examples, ending with the last one, that share parity, multiple-of-5 or final digit (a disconfirming example resets the run). It differs from the incumbent on only 5 sets / 315 trials (e.g. "83,38,58,68" run of 8s, "48,63,53"), so the novelty check might call it a near-duplicate. Smoke fit -2751.4.
**Why:** the brief asked for an order effect; serial-position memory weighting was already pruned. **How to apply:** see [[coincidence-sharpened-exemplar]].
