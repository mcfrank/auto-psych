# Candidate Brief

Refine the incumbent: `prime_over_odd_reading` — rank 0, the best model on this data, ELPD-LOO -2200.5. Its hypothesis, as its author stated it:

> Refinement of `hundred_chart_row_ranges` (everything kept: primitive-built rules with a per-primitive cost, final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment). The one change is that "odd numbers" and "prime numbers" are rival readings of the same kind of evidence (numbers with no factor of 2), and people find "prime" the more meaningful reading and "odd" the emptier, merely "not even" one: a single fitted preference moves prior plausibility away from "odd numbers" and onto "prime numbers" by the same amount. So after all-prime sets (2, 47; 53, 2; 67, 31, 17, 61) people say yes to other primes and no to odd composites, and after all-odd non-prime sets (69, 9, 39, 21) they extend "odd" less than the incumbent predicts and fall back on ranges or narrower rules.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/prime_over_odd_reading.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
