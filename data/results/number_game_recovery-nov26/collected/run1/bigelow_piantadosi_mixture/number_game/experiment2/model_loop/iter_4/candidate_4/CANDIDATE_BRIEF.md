# Candidate Brief

Refine the incumbent: `coincidence_triggered_guess_rethink` — rank 0, the best model on this data, ELPD-LOO -2097.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent digit_coincidence_noticing_guess (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and made salient only once a digit is noticed as shared by at least two examples, fitted trust in the examples). The one change: noticing a coincidence is itself a trigger to rethink. When a new example makes people notice a digit it shares with an earlier example, they drop their current guess with a fitted probability and reconsider the concept from all the examples, even if that guess still fits. So after 46 then 86, a range guess formed from 46 that happens to contain 86 is often given up in favour of "ends in 6", and far numbers like 96 are endorsed more than the purely sticky learner predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/coincidence_triggered_guess_rethink.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
