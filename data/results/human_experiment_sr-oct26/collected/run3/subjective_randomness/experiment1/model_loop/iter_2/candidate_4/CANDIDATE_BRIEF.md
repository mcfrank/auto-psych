# Candidate Brief

Refine the incumbent: `bayesian_overalternating_chance_model` — rank 0, the best model on this data, ELPD-LOO -1021.6. Its hypothesis, as its author stated it:

> People judge randomness as Bayesian inference: a sequence looks random to the extent a fair coin explains it better than a "regular" generator (a coin with an unknown tendency to switch or repeat, or a coin with an unknown bias towards heads or tails), with the regular generators' unknown rates averaged out normatively. The one distortion is in their model of chance itself: people believe a fair coin switches sides more often than half the time, so their "random" likelihood rewards alternation and penalises repeats by a fitted amount — which also sets how much a perfectly alternating sequence is still credited as random rather than condemned as regular.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/bayesian_overalternating_chance_model.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
