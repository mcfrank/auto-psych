---
name: attention-exemplar-similarity
description: exp2 iter_4 c2: GCM exemplar similarity with attention set by example agreement per dimension; smoke LL -2768.6, 3 s sampling
metadata:
  type: project
---
On 2026-10-19, exp2 iter_4 candidate_2 (a similarity brief) submitted `attention_focused_exemplar_similarity`. It is GCM summed exemplar similarity over 10 dimensions (magnitude/10, parity, final digit, decade, divisible by 3/4/5, prime, square, power of 2). Attention is a softmax of focus × pairwise agreement among the examples, plus a magnitude salience term. The yes-rate runs between a floor and a ceiling. Smoke LL was -2768.6, the best similarity account so far (common_feature ~-2843, soft-min ~-2905) but still about 700 nats behind the concept learners. It passed the check in seconds.
**Why:** similarity briefs keep coming back. Precomputed mismatch tensors make these models trivially fast.
**How to apply:** reuse iter_4/candidate_2/candidate.py as the skeleton for later similarity slots. See [[exemplar-softmin-poor-fit]].
