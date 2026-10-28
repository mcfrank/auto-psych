# Candidate Brief

Refine the incumbent: `session_growing_acquiescence_concepts` — rank 0, the best model on this data, ELPD-LOO -2816.2. Its hypothesis, as its author stated it:

> Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100, with evidence weight growing with set size, but as the session wears on they increasingly answer "yes" without consulting their concept at all — a blind-"yes" lapse whose rate rises steadily with how far into the session the trial falls (an inferred starting rate and growth rate). The one change is this session-growing acquiescence, which predicts more "yes" late in the session than early, chiefly to targets the concept clearly excludes (where a blind "yes" makes the most difference), closing the session-liberalization discrepancy the incumbent's critique flags.

Its source is `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/session_growing_acquiescence_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
