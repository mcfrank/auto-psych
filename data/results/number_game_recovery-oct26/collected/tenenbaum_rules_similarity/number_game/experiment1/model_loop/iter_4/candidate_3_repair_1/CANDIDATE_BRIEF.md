# Candidate Brief

Refine the incumbent: `family_pooled_rule_prior` — rank 0, the best model on this data, ELPD-LOO -1247.2. Its hypothesis, as its author stated it:

> Refinement of `decade_bounded_interval_prior`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (rule share, size principle with 10% random examples, Erlang prior over interval width, a preference for decade-bounded ranges, "yes" = posterior probability the concept contains the target), but their prior over rules is set by the kind of rule, not rule by rule. The one change: people hold a belief about how likely each kind of rule is (powers, multiples, final digit, contained digit, and special sets such as primes, evens, odds, squares, cubes) and treat all rules of one kind as equally likely, so "multiples of 7" is as plausible a priori as "multiples of 4", instead of the incumbent's separate free weight for each of the 42 rules.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/family_pooled_rule_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
