# Candidate Brief

Refine the incumbent: `comparison_dependent_concepts` — rank 0, the best model on this data, ELPD-LOO -3110.7. Its hypothesis, as its author stated it:

> Refinement of `digit_rules_need_comparison` (tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals, their odd/even versions and range-bounded rules, with digit rules discounted when only one example is shown). The one change is to say why the discount happens. Any concept that is defined by a relation among the examples can be noticed only by comparing examples. That covers a shared digit pattern and also a rule confined to the span the examples cover ("multiples of 6 between 4 and 9"). So when a single example is shown, range-bounded rules lose prior plausibility by the same inferred amount as digit rules, and no new parameter is added. People should then generalize a lone example to the whole of its arithmetic rule, including far members (6 → 18, 24; 11 → 55, 99; 25 → 100; 10 → 80), more than the incumbent allows, while judgments about sets of two or more examples are unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/comparison_dependent_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
