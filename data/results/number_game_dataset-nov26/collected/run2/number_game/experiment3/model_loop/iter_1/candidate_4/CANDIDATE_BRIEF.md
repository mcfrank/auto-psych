# Candidate Brief

Refine the incumbent: `single_example_random_program_doubt` — rank 0, the best model on this data, ELPD-LOO -4748.4. Its hypothesis, as its author stated it:

> Refinement of the incumbent `span_interval_similarity` by one change. When the program has shown only a single example, people often cannot rule out that it produces numbers more or less at random: on an inferred share of one-example trials they set the example's properties aside and answer at a flat rate that is the same for every target (so some of these answers are a flat "yes" and some a flat "no"), while on the remaining one-example trials and on every multi-example trial they judge exactly as the incumbent does (the salience-weighted share of the examples' shared profile, with the interval similarity). So one-example sets are endorsed more often, and their endorsement depends less on the target's distance and properties than the overlap alone predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/single_example_random_program_doubt.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
