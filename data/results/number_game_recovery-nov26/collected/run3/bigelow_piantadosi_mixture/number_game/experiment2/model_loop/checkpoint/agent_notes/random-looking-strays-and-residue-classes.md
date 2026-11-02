---
name: random-looking-strays-and-residue-classes
description: exp2 iter_2: "strays look random" (anti-familiar) stray density rejected by data (gamma->0); residue-class concepts -70 nats
metadata:
  type: project
---
Exp2 iter_2 cand_1 (2026-11-01). Anchored-reliability base with stray density q(x) ∝ exp(-gamma·standardised familiarity):
MAP gamma=0; observed ll drops 6.5 nats at gamma=0.2, 30 at 0.5. Data favour familiar strays, not random-looking ones. The
familiar_vs_plain_outlier critique (q=0.16) is not strong enough to motivate this direction. Submitted it anyway as random_looking_strays (risk: near-dup of anchored).
Adding residue classes x≡r mod d (r≠0, 5% fixed share) to the anchored model: MAP −70 nats. Ruled out.
Speed trick: for a non-uniform stray density, use a bitmask pattern (first in, later-in bitmask A, size) table and sum over J⊆A as tensordot with an 8x8 subset matrix in prob space (17 ms/grad). logsumexp over a 4-D tensor was 94 ms.
Related: [[familiar-strays-family-level]], [[stray-generator-variants]], [[concept-additions-collapse]]
