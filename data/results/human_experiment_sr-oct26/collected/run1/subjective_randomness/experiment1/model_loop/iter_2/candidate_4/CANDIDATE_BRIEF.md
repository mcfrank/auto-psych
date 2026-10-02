# Candidate Brief

Refine the incumbent: `ideal_alternation_with_balance` — rank 0, the best model on this data, ELPD-LOO -1008.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people additionally expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how it alternates. The one change is this shared penalty on H/T imbalance, added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which the alternation-only model predicts as indifference.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/ideal_alternation_with_balance.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
