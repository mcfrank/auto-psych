# Candidate Brief

Refine the incumbent: `gamblers_run_second_order_side_habit` — rank 0, the best model on this data, ELPD-LOO -3611.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `second_order_motif_side_habit`: people still judge a sequence random to the extent their own second-order model of a fair coin (a person-specific believed switch rate, shifted right after a switch by a shared amount) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length and each person's side habit deciding near-ties. The single change, taken from the narrowly retired `run_length_gamblers_chance_vs_motif`, is that their model of chance also holds a gambler's fallacy: the longer the current run has lasted beyond two flips, the more they expect a fair coin to switch (a shared fitted rise per extra flip), so a long streak looks improbable under chance roughly quadratically in its length — far beyond what its switch count and the after-switch belief imply — while the same switches spread over short runs do not.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/gamblers_run_second_order_side_habit.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
