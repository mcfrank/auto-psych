# Candidate Brief

Refine the incumbent: `coincidence_rule_violation_route` — rank 0, the best model on this data, ELPD-LOO -4083.0. Its hypothesis, as its author stated it:

> Refinement of the incumbent `parity_preserving_decisive_route` (people judge a target by its partly averaged similarity to the examples on magnitude, final digit, parity and the set's common divisor, plus distance-free routes for factor-kin, whole-set round kinship and parity-preserving multiples, and a suspicious coincidence makes the yes/no decision more decisive). The one change: when two or more examples share a suspicious coincidence, people read its most specific form (a shared final digit, else all multiples of 5, else a shared parity) as a rule. A target that breaks that rule loses a fixed share of all its support, its arithmetic and round-number routes included, whatever its distance from the examples. So after "15, 85, 45" or "12, 14, 16", targets that break the rule (e.g. 33 after "12, 14, 16", or 40 after "15, 85, 45") are rejected more consistently than graded similarity allows, even when they are far away or factor-kin of an example. Single examples and incoherent sets are unaffected.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_rule_violation_route.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
