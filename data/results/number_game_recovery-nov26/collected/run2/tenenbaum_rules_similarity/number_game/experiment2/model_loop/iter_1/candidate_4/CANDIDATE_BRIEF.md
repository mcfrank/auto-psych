# Candidate Brief

Refine the incumbent: `without_replacement_example_sampling` — rank 0, the best model on this data, ELPD-LOO -1699.3. Its hypothesis, as its author stated it:

> People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples), but they assume the program never produces the same number twice, so they treat the examples as drawn from the concept without replacement: each further example that fits a concept is taken as one of that concept's members not yet shown. This makes several examples far stronger evidence for a concept that has barely more members than were shown (cubes, powers of 3 or 4, squares, a tight range) than the best model's with-replacement size principle allows, so the sharpest disagreement lives on three- and four-example sets that nearly exhaust a small rule or range, where this model predicts fewer "yes" answers to targets outside that small concept.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/without_replacement_example_sampling.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
