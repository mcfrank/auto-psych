# Candidate Brief

Refine the incumbent: `first_example_anchored_concepts` — rank 0, the best model on this data, ELPD-LOO -1969.8. Its hypothesis, as its author stated it:

> People infer the program's concept by the same Bayesian weighing of mathematical rules and numeric ranges as the best model, but the first example they see anchors that inference: they treat it as the defining instance of the program, so every concept containing the first example gets extra belief, while concepts that would have to dismiss it as a stray number are disfavoured more than concepts that dismiss a later example. The same numbers shown in a different order therefore generalize differently — an odd-one-out shown first pulls belief toward concepts that include it, while the same odd-one-out shown last is readily written off.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/first_example_anchored_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
