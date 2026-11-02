# Candidate Brief

Refine the incumbent: `span_extrapolation_hesitant_guess_2` — rank 0, the best model on this data, ELPD-LOO -3120.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent gamma_width_distance_hesitant_guess (a single sticky working concept guess — a rule or a numerical range with a fitted gamma-shaped width prior — formed from the first example by Bayesian concept learning, digit rules salient once two examples share the digit, coincidences triggering a rethink, fitted trust, evidence-graded misfit tolerance, and reluctance to extend the guessed concept far from the examples). The one change: the reluctance is to extrapolate beyond the region the examples span, not to stray from the nearest example — a target lying between the smallest and largest example is endorsed as fully as concept membership implies, and willingness falls off smoothly (Gaussian-like, fitted reach) only with how far the target lies outside that span. So after 46 and 86, a member such as 66 is not discounted at all, while 96 or 6 are discounted by their distance beyond the span; for a single example the prediction is unchanged. (To keep the fit tractable, the width prior's decay is parameterised by a rate rather than a scale, so a near-flat decay is reached smoothly instead of along a ridge to an infinite scale.)

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/span_extrapolation_hesitant_guess_2.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
