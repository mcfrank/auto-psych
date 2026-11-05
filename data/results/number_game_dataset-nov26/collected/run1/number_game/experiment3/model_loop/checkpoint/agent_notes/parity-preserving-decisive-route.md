---
name: parity-preserving-decisive-route
description: exp3 iter2 c5: parity_preserving_multiple_route + decisive slope graft, MAP +10 over both parents; doesn't fix incoherent polarization
metadata:
  type: project
---
exp3 iter_2 candidate_5 (2026-11-02): parity_preserving_multiple_route with coincidence moved from similarity gain to response slope (incumbent's mechanism). MAP log-lik -4066.0 vs incumbent -4077.6, parity_preserving -4076.0; smoke -4073.5.
Incoherent-cell polarization: predicted 0.137 by both incumbent and this model, against 0.216 observed. The incumbent's predictions bottom out near p≈0.3 for far targets in single-example and incoherent sets (c_mag very shallow).
Things that did NOT fix polarization: a decisiveness term for single examples (s_single≈-0.4, MAP -4059 on top of the graft; singles are *less* decisive), squared distance (-4082), power-exponent distance (shape≈0.44, -4058 on top of the graft, polarization 0.14; on parity_preserving alone -4072, 0.166), log-distance + graft (-4071).
**Why:** follows [[exp3-iter2-critique-stats]], [[coincidence-decisive-exemplar]].
**How to apply:** the polarization gap probably needs target-level structure (e.g. doubles of large odd primes rejected, 14|7 and 99|33 accepted). Set-level decisiveness won't fix it.
