# Candidate Brief

Refine the incumbent: `familiar_strays_equal_rule_prior` — rank 0, the best model on this data, ELPD-LOO -2037.8. Its hypothesis, as its author stated it:

> Refinement of `familiar_strays_family_level_prior`: people do the same Bayesian concept learning over mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes "familiar" numbers (a member of some well-known rule) rather than random ones. The one change is the rule prior: people have no preference among kinds of rule at all — parity, primes, squares, cubes, powers, multiples, final digit and contained digit rules are all equally plausible a priori — so which rule they adopt is decided only by how many examples it explains and how small it is (the size principle), and the rules they would generate a familiar stray from are the same equally weighted rules.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/familiar_strays_equal_rule_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
