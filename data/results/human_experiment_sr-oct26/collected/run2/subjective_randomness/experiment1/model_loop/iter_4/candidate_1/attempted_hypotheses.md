# Tried before

3 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_switch_rate_ideal — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_alternation_ideal' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_alternation_ideal, not a new hypothesis.

**Hypothesis:** Each person carries their own internal ideal of how often a random coin should switch between heads and tails, and judges a sequence as more random the closer its switch rate comes to that personal ideal. People differ in where this ideal sits (some expect heavy alternation, others near-even switching), so the same pair can be judged in opposite directions by different people.

### length_scaled_alternation_ideal_2 — rejected (experiment1 round 2 candidate 4 refine incumbent periodic_penalized_alternation_ideal)

**Outcome:** predicts like existing model 'length_scaled_alternation_ideal' (p_left RMSE 0.00011 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_scaled_alternation_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, with a shared penalty for visibly periodic sequences, but the evidence from switch rate accumulates with the number of transitions people see — a switch-rate deviation observed over seven transitions is a more convincing sign of non-randomness than the same deviation over two. The one change is that the sensitivity to distance from the personal ideal scales as a fitted power of the number of transitions in the sequence, addressing the critique that the model under-predicts how the preference for the more-switching sequence holds up as sequences get longer.

### balance_aware_length_scaled_ideal — rejected (experiment1 round 3 candidate 3 refine incumbent length_scaled_alternation_ideal)

**Outcome:** predicts like existing model 'length_scaled_alternation_ideal' (p_left RMSE 0.00111 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_scaled_alternation_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with sensitivity scaling as a power of the number of transitions) and with a shared penalty for visibly periodic sequences, but people also expect a random coin to give roughly equal numbers of heads and tails, so a lopsided H/T count (e.g. HHHHTT versus HHHTTT, which have the same number of switches) counts against randomness. The one change is a shared penalty on the H/T imbalance of each sequence (the squared deviation of its share of heads from one half, grafted from the global-balance component of `local_representativeness`), addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run — which, at equal switch counts, is the more balanced one.
