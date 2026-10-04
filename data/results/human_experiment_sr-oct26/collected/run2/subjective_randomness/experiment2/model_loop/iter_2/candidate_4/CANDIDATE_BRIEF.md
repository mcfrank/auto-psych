# Candidate Brief

Refine the incumbent: `tally_span_personal_ideal` — rank 0, the best model on this data, ELPD-LOO -2369.5. Its hypothesis, as its author stated it:

> People read a sequence flip by flip while keeping a running tally of how far heads lead tails, and judge randomness by how widely that tally swings over the whole reading — the span between the furthest it ever ran toward heads and the furthest it ever ran toward tails. Each person expects a fair coin's tally to wander over some typical span for a sequence of that length: a long streak (anywhere in the sequence, even one later evened out) stretches the span too far and strict alternation pins it within one step of even, so both look non-random, and people choose the sequence whose tally span is closer to their own expectation. Apart from this judgement, each person has a small habitual lean toward clicking left or right.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/tally_span_personal_ideal.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
