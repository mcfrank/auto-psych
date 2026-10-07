# Candidate Brief

Refine the incumbent: `heavy_tailed_signed_sensitivity_motif` — rank 0, the best model on this data, ELPD-LOO -3568.2. Its hypothesis, as its author stated it:

> Refinement of `attentive_lapse_heads_rigged_motif` (people judge a sequence random to the extent their own second-order model of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a heads-favoured trick coin and a repeating motif with slips — the evidence weighed per flip, with each person's side habit deciding near-ties, and a person-specific share of trials answered at random). The single change is in how people depart from the shared judgment: instead of occasionally guessing at random, which can lower a person's agreement with the majority only down to chance, each person weighs the regularity evidence with their own sensitivity drawn from a heavy-tailed population that can cross zero — most people near the group's typical sensitivity, a few nearly indifferent, and a few reversed, systematically picking the sequence that looks less random to everyone else. This addresses the FDR-surviving critique that participants' agreement with the majority varies across people more than the lapse model produces, since some participants agree with the majority well below chance.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/heavy_tailed_signed_sensitivity_motif.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
