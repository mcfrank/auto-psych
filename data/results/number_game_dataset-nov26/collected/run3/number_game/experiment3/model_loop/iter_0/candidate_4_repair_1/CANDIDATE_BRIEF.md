# Candidate Brief

Refine the incumbent: `consistent_examples_bounded_rules` — rank 0, the best model on this data, ELPD-LOO -4163.9. Its hypothesis, as its author stated it:

> Refinement of `comparison_dependent_concepts` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules and bounded rules discounted when only one example is shown). The one change takes that model's own premise further. A rule confined to the examples' span ("ends in 3 between 48 and 63") is built by comparing the examples and seeing that they share the rule, so people entertain it only when every example fits the rule. A bounded rule that one of the examples breaks is never considered, rather than kept with that example written off as a stray, and no parameter is added. People should then drop a rule that most examples share but one breaks, and fall back on the examples' range or on whole rules. In-range numbers lacking the majority pattern (48,63,53 → 49, 51) should be endorsed more than the incumbent predicts, while sets that a rule fits fully are unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/consistent_examples_bounded_rules.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
