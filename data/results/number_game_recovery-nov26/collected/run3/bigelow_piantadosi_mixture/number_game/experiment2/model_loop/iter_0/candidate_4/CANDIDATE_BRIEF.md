# Candidate Brief

Refine the incumbent: `family_pooled_uncertain_reliability` — rank 0, the best model on this data, ELPD-LOO -2056.6. Its hypothesis, as its author stated it:

> Refinement of `uncertain_reliability_noise_concepts`: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 with the size principle, holding an uncertain belief about how reliable the program is (how often an example is a stray draw from 1-100) and averaging over it for each example set — but their prior over rules is organised by kind of rule rather than set rule by rule. The one change, taken from `family_pooled_rule_prior`, is this family-structured, partially pooled rule prior: each rule's plausibility is drawn around a shared level for its family (parity, primes, squares and cubes, powers, multiples, final digit, contained digit) with an inferred spread, so rules of the same kind (e.g. "contains a 3" and "contains a 7") are about equally plausible unless the evidence says otherwise, which predicts more uniform generalisation across rules within a family than the incumbent's independent rule weights.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/family_pooled_uncertain_reliability.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
