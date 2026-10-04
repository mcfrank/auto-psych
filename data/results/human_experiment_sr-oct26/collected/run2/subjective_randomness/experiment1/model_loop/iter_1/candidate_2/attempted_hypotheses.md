# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_switch_rate_ideal — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_alternation_ideal' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_alternation_ideal, not a new hypothesis.

**Hypothesis:** Each person carries their own internal ideal of how often a random coin should switch between heads and tails, and judges a sequence as more random the closer its switch rate comes to that personal ideal. People differ in where this ideal sits (some expect heavy alternation, others near-even switching), so the same pair can be judged in opposite directions by different people.
