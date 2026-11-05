---
name: property-exemplar-gcm
description: GCM exemplar model over property-family mismatches + magnitude, mean similarity: smoke loglik -3313 (far behind overlap ~-3150)
metadata:
  type: project
---
Exp2 iter_2 cand_0 (exemplar brief): per-example exponential similarity over attention-weighted property-family mismatch counts + |t-e|/10, averaged over examples, sigmoid. Passed check; smoke loglik -3313.2 vs ~-3150 for overlap models. ~34s per 200-iter chain.

**Why:** brief demanded exemplar account; individual-example similarity lacks the consensus/shared-profile emphasis.
**How to apply:** exemplar/prototype slots will likely lose by ~150 nats; don't expect fit gains, see [[final-digit-overshadowing]].
