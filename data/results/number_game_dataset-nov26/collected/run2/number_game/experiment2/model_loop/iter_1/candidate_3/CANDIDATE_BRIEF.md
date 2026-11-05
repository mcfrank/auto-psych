# Candidate Brief

Refine the incumbent: `seed_overshadowing_last_digit_recency` — rank 0, the best model on this data, ELPD-LOO -3194.9. Its hypothesis, as its author stated it:

> Refinement of `seed_multiple_digit_overshadowing_overlap` by one grafted component, taken from `last_example_digit_recency_overlap`: people read the examples in order, and the final digit of the last example shown lingers in their guess as a candidate pattern ("ends in 1") even when the earlier examples do not share it. Otherwise the judgment is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, with "multiples of the smallest example" in that profile when it divides all the others, and a final digit visibly shared by all examples overshadowing every other property. Compared with the seed-overshadowing model, targets that share the last-shown example's final digit are endorsed more than targets that share the first-shown example's, so the same examples in a different order are generalized differently.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/seed_overshadowing_last_digit_recency.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
