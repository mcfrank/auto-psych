# Candidate Brief

Refine the incumbent: `graded_prior_digit_sum_concepts` — rank 0, the best model on this data, ELPD-LOO -3522.8. Its hypothesis, as its author stated it:

> People infer the hidden program as a Bayesian concept learner over number rules and numeric ranges (with family plausibility, the size principle, a preference for moderate range widths, and range outputs clustered toward the range's middle), and their space of candidate programs also includes digit-sum rules — "numbers whose digits add up to s" (e.g. 5, 14, 23, 32, 41, 50) — as a family with its own prior plausibility. Within that family, sums are not equally natural: how plausible a digit-sum rule is a priori changes smoothly with the size of the sum, in a direction and strength inferred from the data. So when the examples share a digit sum, people endorse other numbers with that sum and reject neighbours with a different sum, and they do so more strongly for the favoured sums.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/graded_prior_digit_sum_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
