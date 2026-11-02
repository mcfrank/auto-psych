# Candidate Brief

Refine the incumbent: `variable_min_max_bounded_range` — rank 0, the best model on this data, ELPD-LOO -3323.9. Its hypothesis, as its author stated it:

> Refinement of `min_max_bounded_range`: people do the same Bayesian concept learning over equally plausible mathematical rules, rules held within a magnitude range, and magnitude ranges of 1-100 (size principle, uncertain stray rate, strays sometimes familiar numbers), and they treat the smallest and largest example as the natural limits of a range, so "numbers from the lowest to the highest example" comes to mind far more readily than any other range. The one change, taken from `variable_rule_range_expectation`, is that how strongly people expect the program to follow a rule rather than produce a magnitude range varies from one judgement to the next around an inferred average. So for clustered examples that share a rule (37, 57 or 4, 16, 12), people on some judgements adopt the min-to-max range and fill in the rule's non-members between the examples, never beyond them, and on other judgements they keep the rule.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment3/model_loop/models/variable_min_max_bounded_range.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
