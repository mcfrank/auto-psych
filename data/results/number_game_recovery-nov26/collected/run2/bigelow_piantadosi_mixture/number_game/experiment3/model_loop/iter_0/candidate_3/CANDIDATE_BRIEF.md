# Candidate Brief

Refine the incumbent: `heading_direction_range_extrapolation` — rank 0, the best model on this data, ELPD-LOO -3199.7. Its hypothesis, as its author stated it:

> People read the examples as a sequence and take the direction of the last step (whether the final example went up or down from the one before it) as the direction the program is heading, so the numeric ranges that come to mind as candidate concepts extend further past the examples in that direction and less in the other; otherwise they do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and ranges as the best model. This predicts more "yes" answers to numbers far beyond the examples on the side the sequence was last moving toward (e.g. above 60 after 40, 20, 60) and fewer on the side it moved away from, while rule-based generalization and sets of a single example are unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment3/model_loop/models/heading_direction_range_extrapolation.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
