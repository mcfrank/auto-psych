# Candidate Brief

Refine the incumbent: `uneven_rule_familiarity_ends` — rank 0, the best model on this data, ELPD-LOO -3378.1. Its hypothesis, as its author stated it:

> Refinement of serial_position_ends_inflated_size: people do the same Bayesian concept learning over 42 rules and all ranges of 1-100, with inflated perceived concept sizes, evidence tempered separately for rules and ranges, the first and last examples re-weighted, a rule prior that rises as a power of rule size, and a decision gain with a shifted criterion. The one change is to how familiar each rule seems before any examples: on top of the size trend, each rule has an idiosyncratic familiarity drawn from a wide spread, instead of the incumbent's weights that are pulled toward one another. So people can treat a few rules as strongly familiar ("even numbers", "multiples of 10", "powers of 2") and many others as barely entertained, far more unevenly than the incumbent allows.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/uneven_rule_familiarity_ends.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
