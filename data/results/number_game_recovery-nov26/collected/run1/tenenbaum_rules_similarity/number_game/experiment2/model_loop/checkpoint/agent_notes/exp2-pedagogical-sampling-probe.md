---
name: exp2-pedagogical-sampling-probe
description: Pedagogical (Shafto-style teacher) sampling on threshold incumbent — learned strength goes to 0; fixed full pedagogy loses ~39 nats
metadata:
  type: project
---
iter_2 candidate_1 (2026-10-30): teacher picks x in h with prob ∝ P_L(h|x)^a (learner with the same prior), mixed 10% random. Learned a (HalfNormal or Normal) → 0.012, LL -2245.4 (= incumbent, near-duplicate). Fixed a=1: MAP LL -2284.1 (39 nats worse); flat-prior inner learner with a=1: -2323. Submitted fixed a=1 as pedagogical_teacher_sampling (exploratory).
Speed trick: P_L(h|x) ∝ prior(h)/|h| / Z(x), the prior(h)/|h| cancels in the teacher's normalisation, so teach = M*invZ[x]/(M@invZ)[h]; with count-matrix @ log_example.T and posterior @ M matmuls, 0.09 s/grad for 5302 concepts x 131 sets.
**Why:** pedagogy is ruled out; don't retry the nested version.
**How to apply:** see [[exp2-map-probe-results]] for the other dead ends.

Repair (2026-10-30): the nested version (teacher likelihood depending on the sampled prior) was rejected as too slow (>30 min at target_accept 0.99). Repair fixed the teacher's reference learner at an equal-family prior and Erlang scale 20, so the teacher likelihood is precomputed and grouped by width like the incumbent: smoke fit 200 iters in 107 s, LL about -2358 (around 110 nats worse than the incumbent).
