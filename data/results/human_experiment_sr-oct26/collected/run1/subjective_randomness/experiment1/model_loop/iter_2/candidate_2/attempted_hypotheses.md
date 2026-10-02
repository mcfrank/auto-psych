# Tried before

3 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_switch_rate_prototype — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_ideal_alternation' (p_left RMSE 0.00008 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_ideal_alternation, not a new hypothesis.

**Hypothesis:** Each person carries their own prototype of how often a genuinely random coin switches between heads and tails, and judges a sequence as random to the extent its switch rate is close to that personal prototype. People differ substantially in this prototype (some expect heavy alternation, others expect near-independent switching), so the same pair can be judged in opposite directions by different participants.

### person_prototype_representativeness — rejected (experiment1 round 0 candidate 4 refine incumbent local_representativeness)

**Outcome:** predicts like existing model 'individual_alternation_prototype' (p_left RMSE 0.00063 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of individual_alternation_prototype, not a new hypothesis.

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge a sequence random when it is locally balanced and irregular, but the alternation rate each person's mental prototype of randomness expects differs between people — some expect strong over-alternation, others expect streakier sequences — instead of one shared over-alternating prototype. The single change is that the prototype alternation rate becomes a per-participant parameter (free to lie below or above 0.5) drawn from a population distribution, addressing the critique that real individual differences in alternation preference are far larger than the single-population model produces.

### balanced_personal_ideal_alternation — rejected (experiment1 round 1 candidate 4 refine incumbent personal_ideal_alternation)

**Outcome:** predicts like existing model 'ideal_alternation_with_balance' (p_left RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of ideal_alternation_with_balance, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how often it switches. The single change is this shared H/T-balance penalty (the local-balance component of `local_representativeness`), added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which alternation alone cannot produce.
