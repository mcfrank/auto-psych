---
name: consistency-attention-similarity
description: exp2 iter_1 similarity slot: attention-by-within-set-consistency exemplar model; fast but ll ~-2523 (way behind concept models)
metadata:
  type: project
---
exp2 iter_1 cand_0 (similarity brief): exemplar similarity over magnitude (exp gradient), parity, final digit, shared digit, gcd>=3;
attention = softmax(salience + focus * pairwise within-set agreement). Smoke fit 6 s, ll -2523 vs incumbent ELPD ~-2054.
Similarity-only models keep losing by hundreds of nats (see [[ppc-null-statistics]]); expect pruning.
**How to apply:** in similarity slots, use the prepare_observed padded-distance layout (fast); don't expect it to compete.
