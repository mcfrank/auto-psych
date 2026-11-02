# Candidate Brief

Refine the incumbent: `misfit_tolerant_coincidence_rethink` — rank 0, the best model on this data, ELPD-LOO -3126.9. Its hypothesis, as its author stated it:

> Refinement of the incumbent coincidence_triggered_guess_rethink (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and made salient only once a digit is shared by two examples, a newly noticed coincidence triggering a rethink, fitted trust in the examples). The one change: a guess is not automatically abandoned when a new example contradicts it — people sometimes write the misfitting example off as a stray number and keep their guess, with a fitted tolerance. So after 77 and 17 a guess of "ends in 7" often survives the outlier 8, and targets fitting the early examples' concept are still endorsed despite a later example that breaks it.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/misfit_tolerant_coincidence_rethink.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
