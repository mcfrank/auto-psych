# Candidate Brief

Refine the incumbent: `person_sensitivity_length_scaled_ideal` — rank 0, the best model on this data, ELPD-LOO -1047.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a length-scaled sensitivity and a shared penalty for visibly periodic sequences), but people also differ in how decisively they act on that impression — some pick the sequence nearer their ideal almost every time, others only weakly lean toward it. The one change is a person-specific sensitivity to the alternation-distance, drawn from a population distribution, so that how sharply a person discriminates is an individual trait just as their ideal switching rate is.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/person_sensitivity_length_scaled_ideal.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
