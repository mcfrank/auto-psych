# Candidate Brief

Refine the incumbent: `equal_prior_familiar_rules_digit_sums` — rank 0, the best model on this data, ELPD-LOO -3519.7. Its hypothesis, as its author stated it:

> Refinement of `graded_prior_digit_sum_concepts`: people do the same Bayesian concept learning over number rules, digit-sum rules (with a prior that changes smoothly with the sum) and numeric ranges (Erlang width prior, outputs clustered toward the range's middle, the size principle), but they have no learned preferences among the familiar kinds of arithmetic rule. The one change is to the rule prior: every familiar rule — odd, even, primes, squares, cubes, powers of m, multiples of k, "ends in d" — is equally plausible a priori, with a single inferred share of belief for all of them together (instead of a separate prior mass per family), and ad-hoc rules such as "contains the digit d" or "powers of 2 except 32" are not entertained at all; so how readily a set is read as, say, "multiples of 7" versus "ends in 3" depends only on how well each rule explains the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/equal_prior_familiar_rules_digit_sums.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
