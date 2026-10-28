# Candidate Brief

Refine the incumbent: `rule_size_prior_anchored_evidence` — rank 0, the best model on this data, ELPD-LOO -1945.4. Its hypothesis, as its author stated it:

> Refinement of first_example_evidence_weighting: people do the same Bayesian concept learning over 42 weighted rules and all ranges of 1-100 (evidence tempered separately for rules and ranges, the first example weighed as a stronger or weaker draw than later ones, a decision gain and shifted criterion), but before seeing any examples they do not find tiny, exotic rules ("powers of 4", "powers of 8", "cubes") as plausible as larger, familiar ones ("even numbers", "ends in 4", "multiples of 8"). The one change is that each rule's prior belief rises or falls as an inferred power of how many numbers it contains, which counteracts the size principle for small rules independently of set size, so after 64, 4 people generalize to 14, 24, 44 ("ends in 4", "even") rather than committing to 16 ("powers of 4") and after power-of-2 sets they leak to other even numbers more than the incumbent predicts; an exponent of zero recovers the incumbent.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/rule_size_prior_anchored_evidence.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
