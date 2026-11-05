# Candidate Brief

Refine the incumbent: `progression_graded_magnitude_band` — rank 0, the best model on this data, ELPD-LOO -3168.2. Its hypothesis, as its author stated it:

> Refinement of `gap_step_progression_overlap` by one change, grafted from `graded_magnitude_band_overlap`: people do not treat "at least as large as the smallest example" and "no larger than the largest example" as all-or-none tests, but give a target magnitude credit that falls off smoothly (a bell-shaped gradient of inferred width) with its distance below the smallest or above the largest example. Everything else is unchanged — the salience-weighted share of the examples' shared profile, including the "start at the smallest and add the gap" progression concept, multiples of the seed, final-digit overshadowing and the last example's lingering digit — so the change predicts broader, distance-graded generalization from a single example (near targets endorsed more than far ones), which the incumbent misses.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/progression_graded_magnitude_band.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
