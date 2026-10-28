---
name: uncertain-sampling-sequence-top
description: exp3 iter_1 c5: parity-critique probes null (≤0.6 nats); submitted uncertain_sampling_top_marking with multi-example last-shown top marking
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_5 (refinement slot). The critique parity_match_outside_span_gap (q=0.096) comes mostly from opposite-parity targets outside the span: observed 0.38 vs model 0.44. Same-parity targets fit fine. MAP probes on multi_example_sequence_top_marking: compressed rule size +0.6, a parity-band conj boost +0.1, weak-sampled parity -1.2, dropping parity conjunctions 0. At MAP the conj share is only 0.0065 and the contained-digit kind weight is about 0.
Submitted `uncertain_sampling_sequence_top`: uncertain_sampling_top_marking with tops = 1 iff len>=2 and last==max. MAP LL -3092.6 (parent -3094.0, incumbent -3091.4). MAP p_yes RMSE 0.0056-0.012 against the parents and the incumbent.
Check OK: smoke LL -3095.4, whole check 2m48s.
Generic MAP script: iter_1/candidate_5/scratch/probe/mapgen.py <model.py> <out.npy>.
**Why:** Records that parity mechanisms are exhausted (candidate_3 did parity as its own kind). **How to apply:** Do not re-probe parity tweaks. Reuse mapgen.py as the novelty proxy. Related: [[last-shown-top-discounted-ranges]], [[uncertain-sampling-top-marking]], [[exp3-iter1-critique-stats]]
