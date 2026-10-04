# Candidate Brief

Refine the incumbent: `iter2_candidate3` — rank 0, the best model on this data, ELPD-LOO -2317.6. Its hypothesis, as its author stated it:

> Refinement of the incumbent `tally_span_personal_ideal`: people still keep a running heads-minus-tails tally while reading a sequence and judge it random by how close the tally's span is to their own expected span (with a personal left/right lean), but they also register how often the coin switches sides as a separate sign of randomness, so of two sequences whose tallies swing equally widely they pick the one that alternates more — and people differ in how strongly they reward alternation. The one change is this person-specific preference for the share of H/T switches, addressing the critique (surviving FDR) that among pairs with near-equal tally spans people choose the more-switching sequence far more often than the incumbent predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/iter2_candidate3.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
