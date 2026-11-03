# Candidate Brief

Refine the incumbent: `heading_weber_upward_with_replacement` — rank 0, the best model on this data, ELPD-LOO -3197.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `heading_weber_with_upward_counting`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They allow some strays, sometimes guess, judge a range's breadth Weber-like, let the last step's direction stretch ranges toward where the sequence is heading, and expect the program's numbers to carry on upward past the largest example. The one change is the sampling assumption. People do not assume the program avoids repeating itself, so they read each example as an independent draw from the concept's members (with replacement), not as a draw from the members not yet shown. This predicts that examples that nearly use up a small rule or a short range (e.g. 1, 8, 27 for cubes, or 60, 61, 62, 63 for a four-number stretch) make that concept no more credible than the plain size principle implies, so people say "yes" more often to members of broader rules and ranges after such sets than the incumbent predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment3/model_loop/models/heading_weber_upward_with_replacement.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
