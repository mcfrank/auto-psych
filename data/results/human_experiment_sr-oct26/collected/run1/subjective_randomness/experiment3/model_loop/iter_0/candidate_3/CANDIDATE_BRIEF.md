# Candidate Brief

Refine the incumbent: `most_lopsided_stretch_aversion` — rank 0, the best model on this data, ELPD-LOO -3531.2. Its hypothesis, as its author stated it:

> People notice the single most lopsided stretch of a sequence — the contiguous run of flips in which heads most outnumber tails (or tails most outnumber heads), even when it is not one unbroken streak (HHTHH is a three-heads excess) — and that one striking local excess makes the sequence look less random by a shared weight. This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity to it, personal weights on overall heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate); because a strictly alternating sequence never builds a local excess beyond one flip while a slightly less alternating one with a repeat or two clustered together does, it predicts that among two highly alternating sequences people choose the more alternating one more often than the best model does (the critique's high-alternation discrepancy).

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/most_lopsided_stretch_aversion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
