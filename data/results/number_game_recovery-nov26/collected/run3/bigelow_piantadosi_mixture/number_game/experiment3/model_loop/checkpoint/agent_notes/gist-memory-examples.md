---
name: gist-memory-examples
description: exp3 iter_2 c0: each example verbatim w.p. g else only ±W magnitude gist; exact 16-pattern×16-subset tables; W=10 +1.1 MAP over range_limited, g≈.97; ~27 ms/grad
metadata:
  type: project
---
exp3 iter_2/candidate_0 (2026-11-02), gist_memory_examples, on the range_limited_rule_familiar_strays base (fixed lambda).
- Mechanism: each example is held verbatim with prob g, else only as a gist window [x-W, x+W]. The gist likelihood under concept C is |C∩window|/|C|. The stray mass is the window sum of q.
- Exact, via parameter-free tables per (set, memory pattern P, genuine subset T): sum_C prod_{i∈T} a_i^P(C)/|C|^|T|. In the graph, weight them by c_|T| · prod 1/q and average the per-pattern posterior over pattern probabilities.
- MAP ll: W=10 gives -3319.67, W=20 -3319.77, W=2 -3320.18, incumbent -3320.79. g≈.97 in all cases.
- Speed: rule targets via one GEMM against RULE_MEMBERSHIP (sets·P,100). A sparse CSR one-hot (ps.CSR + structured_dot) replaces the wt[cell_set] gather, whose AdvancedIncSubtensor grad dominated. Even so, 27 ms/grad on 1 CPU. prepare_observed takes ~70 s.
- Gotcha: evaluate p_yes at MAP with model.replace_rvs_by_values, not pytensor.function(value_vars, p_yes), which gives prior draws.
See [[range-limited-rule-familiar-strays]], [[variable-rule-range-expectation]].
