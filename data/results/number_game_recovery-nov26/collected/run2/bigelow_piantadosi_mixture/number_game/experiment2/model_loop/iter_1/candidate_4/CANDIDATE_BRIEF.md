# Candidate Brief

Refine the incumbent: `narrow_range_specificity_bias` — rank 0, the best model on this data, ELPD-LOO -1931.4. Its hypothesis, as its author stated it:

> People infer the hidden concept by Bayesian inference over arithmetic and digit rules and numeric ranges. They read the examples as distinct numbers drawn from the concept, allow that some are stray, and sometimes guess. The one distortion is a specificity bias in how they think about ranges: before seeing any examples, they already judge a narrow stretch of numbers a likelier program than a wide one, so a range's prior plausibility falls (or rises) as a power of its width. The rules themselves keep an equal prior. This predicts tighter "yes" gradients around clustered examples (e.g. 60, 62, 64) and fewer "yes" answers far beyond the examples' span than an unbiased learner gives, while generalization by rules (e.g. powers of 2 after 16, 8, 64) is left unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment2/model_loop/models/narrow_range_specificity_bias.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
