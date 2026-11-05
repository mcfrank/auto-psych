# Candidate Brief

Refine the incumbent: `family_salience_feature_overlap` — rank 0, the best model on this data, ELPD-LOO -1531.5. Its hypothesis, as its author stated it:

> Refinement of `shared_feature_overlap`: people still note every salient mathematical property the examples all share and say "yes" in proportion to how much of that shared profile the target also has, with no size principle. The one change is that the properties are not equally salient: each family of property (parity, primes, squares/cubes, powers, multiples, final digit, contained digit, and lying within the examples' span) gets its own inferred salience weight, so a shared feature of a highly salient kind (e.g. a common final digit) dominates the judgment while shared features of less noticed kinds barely count.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/family_salience_feature_overlap.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
