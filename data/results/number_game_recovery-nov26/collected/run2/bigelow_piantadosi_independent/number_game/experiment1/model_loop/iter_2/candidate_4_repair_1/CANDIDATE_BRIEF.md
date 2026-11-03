# Candidate Brief

Refine the incumbent: `rule_family_prior_tempered_mixture` — rank 0, the best model on this data, ELPD-LOO -888.2. Its hypothesis, as its author stated it:

> Refinement of `tempered_size_principle_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but they do not treat every rule as equally plausible a priori — they hold a prior over *kinds* of rule (parity, primes, squares/cubes, powers, multiples, final digit, contained digit), each kind having its own plausibility, inferred from the data, and the rules within a kind sharing it equally. The single change is this family-structured prior in place of the uniform rule prior: it predicts that whole kinds of rule (e.g. "contains the digit 2" or "multiples of 4") are rarely entertained even when examples fit them, while parity and final-digit rules compete strongly with numerical intervals.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/rule_family_prior_tempered_mixture.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
