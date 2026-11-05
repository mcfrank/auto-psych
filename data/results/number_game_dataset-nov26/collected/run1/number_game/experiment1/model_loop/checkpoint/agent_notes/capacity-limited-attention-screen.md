---
name: capacity-limited-attention-screen
description: iter_4 c2 MLE screen of process variants on shared-factor incumbent; softmax attention budget +2.4 nats submitted, others no gain
metadata:
  type: project
---
iter_4 candidate_2 (2026-11-02), `capacity_limited_attention_exemplar`: the incumbent's magnitude/digit/parity/factor attention became softmax weights that sum to 1, times one sensitivity. MLE NLL 1552.3 vs base 1554.5. check OK, smoke LL -1556.9. Nothing screened moved the critique stats (digit-run nonmatch, mult-5 contrast, set SD). Other results vs base: count-based sequential digit/parity/factor confirmation 1566-1567 (worse), leaky sequential digit run 1552.7, power-mean pooling 1554.3 (pure mean 1568), stochastic example retention 1600, gcd-of-differences skip-count dim 1551-1554, mod-5 residue dim 0 gain.
Data: after 15,85,45 and 55,15,75,45 people endorse 50 at 0.33-0.44 (model 0.18), and after 90,45 they reject 99 (0.11 vs model 0.50).
**Why:** sensitivity and response slope sit on a ridge (product constant), so p uses gain*L/sens. **How to apply:** keep that parameterisation and mask padded slots in screens. Bounded L-BFGS from base params beats Nelder-Mead, which diverged. See [[iter4-critique-stats]].
