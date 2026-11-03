---
name: minerva-echo-concept-traces
description: exp3 iter_4 c1 exemplar slot: MINERVA-2 echo over rule/range traces (activation = cosine^p) — ll −4284, ~960 nats worse; echo retrieval ruled out
metadata:
  type: project
---
exp3 iter_4/candidate_1 (2026-11-02) is minerva_echo_concept_traces. The examples act as a probe. Each of the 45 rules and 5050 ranges is a trace, activated by prior × (k/sqrt(|C| n))^p. p_yes is the share of the echo coming from traces that contain the target. It has 2 params.
- At the MAP, p = 2.65, lambda = .40 and ll = −4284 (the concept models reach about −3325). Most predictions fall between .1 and .5, so they are far too diffuse. Activation grows only by a fixed power of overlap, so it cannot reproduce the |C|^-n tightening of the size principle.
- Speed trick: group ranges by (size, overlap k) counts per set and per cell, so the graph works on a (sets, 100, 5) array. 200 NUTS iterations take 35 s.
**How to apply:** do not propose exemplar or echo accounts that lack a multiplicative per-example likelihood. A cosine or power-of-overlap similarity is ruled out. See [[graded-exemplar-equal-rule-prior]], [[ratio-scale-exemplar-retrieval]].
