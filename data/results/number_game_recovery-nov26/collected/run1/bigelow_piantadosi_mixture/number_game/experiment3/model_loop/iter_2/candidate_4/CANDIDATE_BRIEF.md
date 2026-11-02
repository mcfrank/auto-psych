# Candidate Brief

Refine the incumbent: `gamma_width_range_prior_guess` — rank 0, the best model on this data, ELPD-LOO -3123.8. Its hypothesis, as its author stated it:

> Refinement of the incumbent evidence_graded_misfit_tolerance (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and a contradicting example written off as a stray more readily the more earlier examples the guess explains). The one change: people's prior sense of how wide the program's range of numbers is does not have the fixed Erlang form the incumbent assumes, but a gamma form whose shape is fitted, so how implausible very narrow ranges hugging the examples are, relative to how fast very broad ranges become implausible, is set by the data. With a shape below the incumbent's, tight ranges around clustered examples are more readily entertained while broad ranges stay plausible longer, so people generalise sharply right around clustered examples but with a longer tail out to distant numbers.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/gamma_width_range_prior_guess.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
