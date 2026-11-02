---
name: exp2-simplicity-prior-probe
description: Exp2 iter_4 one-parameter description-length prior in place of family masses + Erlang loses ~875 nats
metadata:
  type: project
---
iter_4 candidate_0 (2026-11-01, exploratory "simpler" brief): without-replacement concept learning with the prior set only by exp(-cost * #symbols). Symbol counts: odd/even/primes/squares/cubes 1; multiples/powers/ends/contains 2; threshold 2; bounded range 3; exception rules 4. Bounded ranges are flat in width. MAP cost 2.29, LL -3120, against about -2245 for the incumbents. Variant where numeric arguments are coded uniformly (prior / #choices): cost goes to 0, LL -3289. Submitted v1 as simplicity_description_length_prior; smoke check OK and fast.
**Why:** learned family masses (contains-digit and exceptions at 0, ranges 0.55) plus the Erlang width prior carry ~875 nats; a single simplicity bias cannot reproduce them.
**How to apply:** don't retry fixed or structured family priors; see [[exp2-map-probe-results]], [[exp2-iter3-saturation-probes]].
