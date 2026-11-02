# Candidate Brief

Refine the incumbent: `example_anchored_exception_neighbourhood` — rank 0, the best model on this data, ELPD-LOO -2642.2. Its hypothesis, as its author stated it:

> Refinement of `rule_plus_exception_neighbourhood`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus a small fuzzy neighbourhood" concepts (radius zero to three, smaller radii a priori likelier), but they build those exceptions bottom-up from what they saw: a rule's exception region is only ever centred on an example that the rule fails to cover ("ends in 2, and also numbers around 27" because 27 was shown), never at an arbitrary place on the number line, and these anchored variants share a fitted part of the rule's prior. The one change is this example-anchored proposal of exceptions, replacing the incumbent's uniform prior over every possible exception location: when one example breaks the others' pattern, the rule-plus-neighbourhood account gets far more belief and is centred squarely on the outlier, so people endorse the outlier's close neighbours, falling off with distance from it, much more than mere neighbours of the majority examples that break the rule (closing the critique's outlier-versus-majority neighbour contrast).

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment3/model_loop/models/example_anchored_exception_neighbourhood.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
