---
name: nearest-example-proximity-pull
description: exp3 iter_3 c1: beta_share + logit pull strength·exp(-d_nearest/tau); smoke OK ll −3654.1; dup risk if strength≈0
metadata:
  type: project
---
exp3 iter_3 candidate_1 (2026-10-31), nearest_example_proximity_pull: the similarity-brief answer built on the beta_share learner (copied from models/beta_share_width_exception_concepts.py). Adds strength ~ N(0,1) and tau ~ LogNormal(log 4, 0.7) and computes per-cell distance to the nearest example in prepare_observed. Smoke OK, 20 s, ll −3654.1, about the base's level. The strength posterior and novelty were not checked.
**Why:** pure similarity models lost 120–900 nats ([[exemplar-concept-similarity-pooling]], [[graded-neighbourhood-magnitude]]), so similarity was added only as a readout pull.
**How to apply:** if it is rejected as a duplicate of beta_share, then people in exp3 show no proximity bias beyond what ranges already give.
