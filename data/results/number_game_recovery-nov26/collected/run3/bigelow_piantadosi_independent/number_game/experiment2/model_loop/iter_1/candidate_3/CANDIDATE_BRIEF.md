# Candidate Brief

Refine the incumbent: `factor_graded_multiples_concepts` — rank 0, the best model on this data, ELPD-LOO -2304.2. Its hypothesis, as its author stated it:

> Refinement of `costly_squares_final_digit_concepts`. Everything is kept: concepts built from mental primitives with a fitted cost per extra primitive, a discount for "ends in d" rules, separate extra costs for powers/cubes and for perfect squares, Erlang-weighted ranges, the size principle and near-all-or-none commitment. The one change is that "multiples of m" rules are not equally available for every m. People check divisibility by recalling a times table, and larger tables are less fluent, so the prior cost of a multiples rule grows with the logarithm of its factor, at a fitted rate. People therefore readily read sets as multiples of 3, 4 or 5 (e.g. 20, 32, 92 as multiples of 4; 90, 100, 5 as multiples of 5). After 14, 98, 91 or 8, 64 they seldom commit to "multiples of 7" or "multiples of 8" and say no to other multiples (7, 35, 63; 24, 40, 56).

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/factor_graded_multiples_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
