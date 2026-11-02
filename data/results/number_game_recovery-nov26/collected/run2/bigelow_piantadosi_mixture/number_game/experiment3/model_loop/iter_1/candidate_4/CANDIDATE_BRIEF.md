# Candidate Brief

Refine the incumbent: `heading_extrapolation_weber_width` — rank 0, the best model on this data, ELPD-LOO -3198.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent `heading_direction_range_extrapolation`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays and sometimes guess, and they let the direction of the last step stretch the ranges they consider toward where the sequence is heading. The one change is in how they judge a range's breadth before any examples. They judge it Weber-like, by its width relative to the magnitude of the numbers it covers, not by its absolute width. So "60 to 70" counts as a tighter, more specific range than "2 to 12", and the strength of this preference is inferred. This predicts that generalization around large clustered examples falls off differently from generalization around small ones, while the heading effect, rule and digit-sum generalization are kept.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment3/model_loop/models/heading_extrapolation_weber_width.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
