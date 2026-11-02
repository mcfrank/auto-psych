# Candidate Brief

Refine the incumbent: `gamma_width_exception_family_concepts` — rank 0, the best model on this data, ELPD-LOO -2291.2. Its hypothesis, as its author stated it:

> Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/gamma_width_exception_family_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
