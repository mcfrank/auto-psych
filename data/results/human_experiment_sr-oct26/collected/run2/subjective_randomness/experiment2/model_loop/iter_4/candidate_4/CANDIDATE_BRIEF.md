# Candidate Brief

Refine the incumbent: `tally_span_switch_rate_ideal_2` — rank 0, the best model on this data, ELPD-LOO -2242.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `iter2_candidate3`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (with person-specific, length-scaled sensitivity and a personal left/right lean), and still attend to how often the coin switches sides with person-specific strength — but switching is no longer rewarded "the more the better": people expect a random coin to switch at a particular rate (somewhat above one half), so a sequence that switches less than that looks streaky and one that switches more, up to strict alternation like HTHTHTHT, looks too regular. The one change replaces the incumbent's linear switching reward with a closeness-to-an-ideal-switching-rate judgement (a shared ideal rate, person-specific weight), addressing the critiques (surviving FDR) that people pick a perfect alternator far less often than the linear reward implies and that the alternation preference changes shape among high-switching pairs.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/tally_span_switch_rate_ideal_2.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
