# Candidate Brief

Refine the incumbent: `decimal_multiple_salience_overlap` — rank 0, the best model on this data, ELPD-LOO -1508.1. Its hypothesis, as its author stated it:

> Refinement of `family_salience_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to the salience-weighted share of that profile the target also has, with no size principle. The one change is that multiples of 5 and of 10 — the round, base-ten multiples — form their own salience family, separate from other multiples (of 3, 4, 6, 7, 8, 9, 11, 12), because decimal notation makes them far more noticeable; so when examples like 90 and 45 share both "multiple of 5" and "multiple of 9", people generalize along the multiple-of-5 property much more than along the other one.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/decimal_multiple_salience_overlap.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
