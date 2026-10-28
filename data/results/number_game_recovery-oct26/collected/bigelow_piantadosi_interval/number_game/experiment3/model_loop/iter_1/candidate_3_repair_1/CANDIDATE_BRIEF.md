# Candidate Brief

Refine the incumbent: `inflated_size_familiar_rule_prior` — rank 0, the best model on this data, ELPD-LOO -3425.1. Its hypothesis, as its author stated it:

> Refinement of inflated_concept_size_principle: people do the same Bayesian concept learning over 42 weighted rules and all ranges of 1-100 (evidence tempered separately for rules and ranges, a stronger- or weaker-weighted first example, a decision gain and shifted criterion), and in the size principle they still perceive every concept as larger than it really is by an inferred constant number of members. The one change, taken from rule_size_prior_anchored_evidence, is that even before seeing any examples they find small, exotic rules ("powers of 4", "powers of 8", "cubes") less plausible than large, familiar ones ("even numbers", "ends in 4", "multiples of 8"), with each rule's prior belief rising as an inferred power of how many numbers it contains. This preference is the same whatever the number of examples, while the perceived-size bias matters more the more examples are shown. So after sets where a tiny rule sits inside a larger one (8, 64; 64, 4; 2, 8; 16, 32, 2, 8), people generalize to the larger rule's members more than either source model predicts, at every set size; an exponent of zero recovers inflated_concept_size_principle.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/inflated_size_familiar_rule_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
