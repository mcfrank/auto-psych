# Candidate Brief

Refine the incumbent: `upper_edge_marking_ranges` — rank 0, the best model on this data, ELPD-LOO -2060.3. Its hypothesis, as its author stated it:

> Refinement of `range_restricted_rule_concepts` (with its prime concept corrected to exclude 1, as in the incumbent): people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules, but they assume the program's examples mark where its range stops at the top — the largest example shown is taken as evidence that the range ends there. The one change, grafted in one-sided form from `endpoint_marking_interval_sampling`: a range whose upper end coincides with the largest example gets an inferred evidence bonus, while its lower end is unconstrained, so people generalize downward from the examples more readily than upward (after 16 they accept 15 or 13 more than 17 or 19; after 83,77 they reject 84).

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/upper_edge_marking_ranges.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
