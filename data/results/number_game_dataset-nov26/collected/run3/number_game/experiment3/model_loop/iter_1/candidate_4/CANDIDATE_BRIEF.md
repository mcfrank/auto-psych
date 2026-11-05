# Candidate Brief

Refine the incumbent: `range_specific_evidence_strength` — rank 0, the best model on this data, ELPD-LOO -4152.4. Its hypothesis, as its author stated it:

> Refinement of `consistent_examples_bounded_rules`. People use the same tempered, sample-size-neglecting Bayesian concept learning over weighted rules, size-weighted ranges, their odd/even versions and bounded rules that every example must fit, and they still discount comparison-based rules when shown one example. The one change: people do not weigh the examples as equally strong evidence for every kind of concept. Once they think of a magnitude range, the size principle acts on it with its own strength, which differs from the strength used for mathematical and digit rules by an inferred amount (no difference recovers the incumbent). This separates how tightly the imagined range fits the examples from how sharply people choose among rules, so it should change how much people endorse in-range relative to out-of-range targets for pairs of examples and for sets that no rule fits fully, while judgments driven by a rule that fits fully stay nearly the same.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/range_specific_evidence_strength.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
