# Candidate Brief

Refine the incumbent: `evidence_graded_misfit_tolerance` — rank 0, the best model on this data, ELPD-LOO -3125.9. Its hypothesis, as its author stated it:

> Refinement of the incumbent misfit_tolerant_coincidence_rethink (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and a contradicting example sometimes written off as a stray). The one change: how readily people write off a contradicting example grows with how much evidence their guess already has — each earlier example the guess explains is an independent reason to keep it, so the chance of dismissing the misfit is one minus the chance that none of those confirmations holds (the same fitted tolerance per confirming example, zero for a guess that explains nothing). So "ends in 7" survives an outlier 8 more often after 77, 17, 37 than after 77, 17, and a misfit late in a long set is shrugged off more than one early in a short set. (To keep the fit tractable, the incumbent's power-law preference for where a range lies on the number line, which earlier rounds found the data barely constrain, is fixed at neutral.)

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/evidence_graded_misfit_tolerance.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
