---
name: log-ratio-nearest-similarity
description: exp3 iter3 log-ratio nearest-example similarity (2 params) smoke loglik -4480, ~1000 behind incumbent
metadata:
  type: project
---
exp3 iter_3 cand_1 (similarity brief): sigmoid(a - b*min|log t - log x|). Smoke loglik -4480.1 vs incumbent ELPD ~-3485. Pure magnitude similarity is far worse than even MINERVA echo (-4283) — see [[minerva-echo-exemplar]].
**Why:** similarity briefs keep recurring; pure-magnitude metrics ignore rule structure.
**How to apply:** if a similarity brief returns, use property-based metrics, not magnitude alone.
