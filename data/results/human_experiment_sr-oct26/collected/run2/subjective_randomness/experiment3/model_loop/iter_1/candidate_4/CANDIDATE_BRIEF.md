# Candidate Brief

Refine the incumbent: `rule_built_tally_switch_triplet` — rank 0, the best model on this data, ELPD-LOO -3413.6. Its hypothesis, as its author stated it:

> Refinement of `tally_span_switch_ideal_triplet_variety` (running heads-minus-tails tally judged by closeness of its span to a personal expected span with person-specific, length-scaled sensitivity; switching judged by closeness to a shared ideal rate with person-specific weight; a reward for variety of three-flip chunks; a personal left/right lean): people also notice when a whole sequence has been built by a simple construction rule, and see it as designed rather than random. The rule can be copying a short unit (HTHTHTHT, HHTHHT, HTTTHTTT) or a reversed mirror image with heads and tails swapped (HHHTHTTT, HHHHTTTT, HTTHTHHT read backwards with faces swapped gives itself). The one change is a single shared penalty on rule-built sequences of either kind. Such symmetric sequences are not periodic and their chunk variety is ordinary, which is why the current models over-predict how often people choose them.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/rule_built_tally_switch_triplet.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
