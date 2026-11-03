# Candidate Brief

Refine the incumbent: `concepts_sampled_without_replacement` — rank 0, the best model on this data, ELPD-LOO -907.2. Its hypothesis, as its author stated it:

> Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with a size-graded rule prior, 10% of examples random and occasional guessing at their own yes rate. They also take it that the program shows distinct numbers, so each further example is a draw from the concept's members not yet shown (strong sampling without replacement). The one change is this sampling assumption. It sharpens the size principle most for small concepts that are nearly used up by the examples (e.g. 8, 64 or 16, 8, 64 strongly favour "powers of 2", or 1, 8, 27 "cubes"), so people say "yes" less often to members of broader rules and wide ranges than a with-replacement learner predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/concepts_sampled_without_replacement.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
