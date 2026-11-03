# Candidate Brief

Refine the incumbent: `digit_coincidence_noticing_guess` — rank 0, the best model on this data, ELPD-LOO -2097.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent sticky_guess_fitted_sampling_trust (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples). The one change: a rule defined only by how numbers are written ("ends in 6", "contains the digit 3") becomes a salient candidate only once people notice that digit as a coincidence shared by at least two of the examples seen so far; until then — always after a single example, and for a digit only one example has — its plausibility is reduced by a fitted availability factor. So a lone 46 does not readily lead people to say yes to 96, whereas 46 and 86 together do.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/digit_coincidence_noticing_guess.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
