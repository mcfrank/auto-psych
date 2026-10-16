# Candidate Brief

Refine the incumbent: `second_order_motif_side_habit` — rank 0, the best model on this data, ELPD-LOO -2103.4. Its hypothesis, as its author stated it:

> Refinement of the incumbent `second_order_chance_motif_suspicion`: people still judge a sequence random to the extent their own second-order model of a fair coin (a person-specific believed switch rate, and a shared shift in how likely a switch is right after a switch) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length. The single change, taken from `person_side_habit_switch_belief`, is that each person also has their own habitual leaning towards the Left or the Right button, drawn from a population, which adds to the randomness evidence before the choice, so when the two sequences look about equally random a person's side habit decides — addressing the FDR-surviving critique that participants' proportions of Left choices vary more across people than the incumbent produces.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/second_order_motif_side_habit.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
