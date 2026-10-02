# Candidate Brief

Refine the incumbent: `personal_lapse_ideal_alternation` — rank 0, the best model on this data, ELPD-LOO -973.3. Its hypothesis, as its author stated it:

> People judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but the decision rule is not always engaged: on some trials a person does not compare the sequences at all and picks a side by guessing. How often this happens is a stable trait that differs between people, so some participants follow their alternation preference almost every trial while others are close to indifferent, which spreads individual choice proportions beyond what a single shared decisiveness produces.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/personal_lapse_ideal_alternation.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
