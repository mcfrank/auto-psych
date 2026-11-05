# Candidate Brief

Refine the incumbent: `monotone_order_range_prior_shift` — rank 0, the best model on this data, ELPD-LOO -1549.4. Its hypothesis, as its author stated it:

> Refinement of `serial_position_weighted_evidence`: people do the same tempered, serial-position-weighted Bayesian concept learning over the 45 individually weighted rules and all intervals of 1-100, but when three or more examples are shown in sorted (steadily rising or steadily falling) order, they read the list as a progression along the number line and so shift prior belief from mathematical/digit rules toward magnitude-range concepts, by an inferred amount. The one change is this order-induced shift in the rule-versus-range prior (zero shift recovers the original model); it predicts that a sorted list such as 62, 32, 22 loosens the "ends in 2" rule, so in-range or nearby numbers lacking the shared digit are endorsed more than when the same kind of set is shown in scrambled order (e.g. 63, 23, 43).

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/monotone_order_range_prior_shift.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
