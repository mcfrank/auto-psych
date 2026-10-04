# Candidate Brief

Refine the incumbent: `tally_span_switch_ideal_periodic_unit` — rank 0, the best model on this data, ELPD-LOO -3435.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `tally_span_switch_rate_ideal_2`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, a personal left/right lean), and still judge switching by closeness to a shared ideal switching rate with a person-specific weight — but they also spot a short unit being repeated (strict alternation HTHT..., or motifs like HHTHHT and HTTTHTTT, any unit of two or more flips repeated at least twice through the whole sequence), and a sequence with such a visible repeating pattern looks designed rather than random. The one change is this shared penalty on visible periodicity (taken from the alternation-ideal models), which lets the switching ideal sit higher so that heavy switching is not over-penalised, addressing the critiques (surviving FDR) that people choose periodic period-3/4 motifs less often, and the more-switching sequence among high-switch pairs more often, than the incumbent predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_span_switch_ideal_periodic_unit.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
