# Candidate Brief

Refine the incumbent: `short_sequence_pseudoflip_dilution` — rank 0, the best model on this data, ELPD-LOO -3566.8. Its hypothesis, as its author stated it:

> Refinement of the incumbent `heavy_tailed_signed_sensitivity_motif` (people judge a sequence random to the extent their own second-order model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a heads-favoured trick coin and a repeating motif with slips — weighing the evidence per flip, with heavy-tailed signed person sensitivity and a side habit). The single change is in how the evidence is weighed per flip: people do not judge a very short sequence on its few flips alone but read its impression as if diluted by a handful of imagined, unremarkable flips (a fitted pseudo-length added to the real length), so the per-flip weighting no longer inflates the evidence of 2- and 3-flip sequences — people still prefer the more-switching sequence in very short pairs, but less decisively than a pure per-flip scaling implies, addressing the critique that in pairs of length 2–3 people choose the more-switching sequence less often than the incumbent predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/short_sequence_pseudoflip_dilution.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
