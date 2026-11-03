# Candidate Brief

Refine the incumbent: `exception_rule_family_prior_2` — rank 0, the best model on this data, ELPD-LOO -928.6. Its hypothesis, as its author stated it:

> Refinement of `family_weighted_concept_prior`: people do the same Bayesian concept learning over rules and intervals with the size principle and a separate prior mass per concept family, but the two gerrymandered exception rules ("powers of 2 except 32" and "powers of 2 plus 37") are not natural mathematical sets — people give them their own, separately inferred (and expectedly tiny) prior mass rather than an equal share of the "special sets" family alongside primes, squares and powers. The one change is this separate family for exception rules, so for sets of powers of 2 people should reject the lure 37 nearly always while still endorsing the other powers of 2.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment1/model_loop/models/exception_rule_family_prior_2.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
