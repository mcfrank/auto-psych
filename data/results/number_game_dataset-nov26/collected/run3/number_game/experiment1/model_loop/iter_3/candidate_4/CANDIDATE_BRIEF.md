# Candidate Brief

Refine the incumbent: `parity_restricted_concepts` — rank 0, the best model on this data, ELPD-LOO -1553.4. Its hypothesis, as its author stated it:

> Refinement of `tempered_size_principle_independent`: people do the same tempered-evidence Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but parity is a basic feature they can conjoin with any concept, so they also entertain the odd-only and even-only version of each rule and interval (e.g. "odd multiples of 3", "even numbers between 4 and 96"), which together receive an inferred share of prior belief. The one change is adding these parity-restricted concepts (share 0 recovers the incumbent); because such a concept is half the size, the size principle favours it whenever all examples share a parity, so people reject opposite-parity targets — including far, weakly supported ones outside the examples' range — more firmly than the incumbent predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/parity_restricted_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
