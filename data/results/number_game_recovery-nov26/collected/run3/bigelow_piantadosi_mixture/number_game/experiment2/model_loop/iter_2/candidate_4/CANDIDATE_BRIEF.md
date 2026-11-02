# Candidate Brief

Refine the incumbent: `familiar_strays_family_level_prior` — rank 0, the best model on this data, ELPD-LOO -2051.2. Its hypothesis, as its author stated it:

> Refinement of `familiar_number_strays`: people do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, an uncertain belief about how often an example is a stray), and they think of a stray as sometimes a "familiar" number (a member of some well-known rule) rather than always a random number, so examples that many small rules produce (1, 2, 4, 8, 16, 64, 100) are easily written off while unremarkable ones (58, 87, 94) must be covered by the concept. The one change, taken from `first_example_anchored_reliability`, is the rule prior: people judge a rule's plausibility only by its kind (parity, primes, squares and cubes, powers, multiples, final digit, contained digit), so all rules of a kind are equally plausible, with no rule-specific preferences — predicting the same generalisation for, e.g., "contains a 3" and "contains a 7" whenever the examples fit them equally.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/familiar_strays_family_level_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
