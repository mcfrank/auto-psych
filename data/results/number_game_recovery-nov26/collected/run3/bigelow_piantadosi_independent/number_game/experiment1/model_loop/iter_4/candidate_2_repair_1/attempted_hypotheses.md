# Tried before

3 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### divisibility_salient_simplicity_search — rejected (experiment1 round 3 candidate 3 refine incumbent erlang_interval_simplicity_search)

**Outcome:** predicts like existing model 'erlang_interval_simplicity_search' (p_yes RMSE 0.00086 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_simplicity_search, not a new hypothesis.

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, the size principle, an Erlang size prior on intervals, near-all-or-none commitment to the best-explaining concept). The one change is that divisibility — "multiples of m" — is a privileged, highly accessible kind of arithmetic rule whose prior is shifted by its own fitted salience bonus, rather than costing exactly what every other two-primitive rule (powers, final digits, containing a digit, squares) costs. If divisibility is salient, a set whose examples all share a factor (e.g. 18, 45, 72 or 39, 60, 12, 45) is read as "multiples of m" more often than as a power, a final digit or a range, so people say yes to other multiples of m more than the incumbent predicts, as the critique reports.

### divisibility_salient_concept_search — rejected (experiment1 round 3 candidate 4 refine incumbent erlang_interval_simplicity_search)

**Outcome:** predicts like existing model 'erlang_interval_simplicity_search' (p_yes RMSE 0.00093 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_simplicity_search, not a new hypothesis.

**Hypothesis:** Refinement of `erlang_interval_simplicity_search`: people still build concepts from mental primitives (a fitted cost per extra primitive, Erlang-weighted intervals, the size principle and near-all-or-none commitment to the best-explaining concept). The one change is that divisibility is a privileged, readily accessible kind of rule: "multiples of m" concepts receive a fitted prior bonus over other two-primitive rules (powers, ending in a digit, containing a digit), so when all the examples share a divisor people more often commit to a multiple-of-m rule and say yes to other multiples of it — closing the critique that people extend arithmetic rules more than the incumbent predicts.

### hidden_digit_costly_concepts — rejected (experiment1 round 4 candidate 1 lens 2)

**Outcome:** predicts like existing model 'contains_digit_costly_concepts' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of contains_digit_costly_concepts, not a new hypothesis.

**Hypothesis:** People's number concepts are built from multiplication (parity, multiples of m, final digits, ranges, with powers/squares/cubes costing extra), but a digit buried anywhere inside a numeral is hard to notice as the basis of a rule: concepts of the form "contains the digit d" carry a fitted extra prior cost over other rules of the same length. So when all examples happen to contain a 2 (e.g. 92, 82, 27, 42), people read them as "ends in 2" with a stray, or as a range, rather than "contains a 2", and say no to targets such as 23 or 25 that share a contained digit but not a final digit.
