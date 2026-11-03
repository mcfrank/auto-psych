---
name: normalized-attention-similarity
description: exp2 similarity family best form — normalised attention budget over shared properties vs magnitude; MAP ~2636, smoke loglik -2627.7
metadata:
  type: project
---
Exp2 iter_4 cand_2 (exploratory, similarity brief): attention budget split between shared-property surprise ((n-1)*-log base rate) and a magnitude salience that shrinks with log spread; distance = violated share; sigmoid. MAP NLL ~2636 vs diagnostic_attention ~2683 vs incumbent ~2150. Unnormalised version ~2678; Weber/log distance worse than linear; distance outside span beats nearest-example distance. Lapse fits to ~0.

**Why:** similarity models stay ~480 nats behind concept learners; normalisation is the biggest gain within the family.
**How to apply:** if a similarity brief recurs, start from this form; related [[attention-exemplar-finding]].
