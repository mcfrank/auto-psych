# Candidate Brief

Refine the incumbent: `streak_weary_chance_pseudoflip` — rank 0, the best model on this data, ELPD-LOO -3565.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `short_sequence_pseudoflip_dilution` (people judge a sequence random to the extent their own second-order picture of a fair coin explains it better than the regular generators they suspect — a switch-biased coin, a heads-favoured trick coin and a repeating motif with slips — with heavy-tailed signed person sensitivity, a side habit, and the per-flip evidence diluted by imagined pseudo-flips for short sequences). The single change, taken from `gamblers_run_second_order_side_habit`, is to their picture of chance: people hold a gambler's fallacy that a fair coin becomes more likely to switch the longer the current streak has lasted, so a long unbroken streak is improbable under chance far beyond its switch count and the first one or two breaks of a long streak make a sequence look much more random — addressing the FDR-surviving critique that among long, low-switch pairs people choose the sequence with more streak breaks more often than the incumbent predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/streak_weary_chance_pseudoflip.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
