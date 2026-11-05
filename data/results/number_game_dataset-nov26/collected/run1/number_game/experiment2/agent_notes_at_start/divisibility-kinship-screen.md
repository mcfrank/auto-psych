---
name: divisibility-kinship-screen
description: iter_3 cand_4 MAP screen on graded-parity incumbent — coprime penalty -7.5 nats best; digit-homog forms ~-5; criterion shifts ≤4
metadata:
  type: project
---
iter_3 candidate_4 (2026-11-02), MAP with the incumbent's priors (base data NLL 1562.2): adding a coprime(t,e) similarity penalty on top of parity gives 1554.7 (submitted as `divisibility_kinship_exemplar`, smoke log-lik -1558.2). Using coprime *instead of* parity is worse (1567.4); set-GCD non-divisor penalty 1560.5; digit attention driven by an all-share indicator or homog^8 gives 1556.7; counting shared-digit examples (not fraction) is much worse (1572.7); set-level criterion shifts (digit homog 1557.9, parity homog, spread, n examples) give ≤4; normalizing by the similarity mass over 1-100 gives nothing. A per-set random intercept gains a lot in-sample but would give the same predictions as the incumbent on unseen sets, so the novelty gate would likely reject it.

**Why:** the sets that miss most are rule-like: 3,63→30/36/39 get a lot of yeses; 46,76,66→64/68 and 78,48,88→80-86 get almost none.
**How to apply:** next, try strict digit (all-share) attention on top of divisibility kinship. See [[iter3-critique-stats]], [[fixed-parity-incumbent-repair]].
