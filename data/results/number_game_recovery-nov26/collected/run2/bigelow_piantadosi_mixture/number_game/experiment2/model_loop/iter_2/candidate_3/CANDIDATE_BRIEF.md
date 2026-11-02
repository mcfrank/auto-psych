# Candidate Brief

Refine the incumbent: `digit_sum_concepts` — rank 0, the best model on this data, ELPD-LOO -1930.2. Its hypothesis, as its author stated it:

> People do Bayesian concept learning over the same arithmetic and digit rules and numeric ranges as the best model: distinct examples drawn without replacement, some strays, occasional guessing, and a prior that favours narrow ranges. They also read the numbers as written numerals and entertain one more kind of concept, "numbers whose digits add up to s" (for example 18, 27, 36, 45 or 13, 22, 31, 40). All 18 such digit-sum concepts are equally likely a priori, and the family as a whole takes an inferred share of the prior belief given to rules. This predicts that examples sharing a digit sum (e.g. 36, 45, 54, 81) lead to "yes" for other numbers with the same digit sum, such as 63, 72, 90 or 27, beyond what "multiples of 9" or a range gives, and to fewer "yes" answers for other members of those broader concepts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment2/model_loop/models/digit_sum_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
