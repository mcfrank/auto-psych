# Candidate Brief

Refine the incumbent: `variable_rule_range_expectation` — rank 0, the best model on this data, ELPD-LOO -3324.3. Its hypothesis, as its author stated it:

> Refinement of `range_limited_rule_familiar_strays`. People do the same Bayesian concept learning over equally plausible mathematical rules, rules held within a magnitude range, and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes familiar numbers. The one change is that how strongly people expect the program to follow a mathematical rule rather than produce a magnitude range is not fixed: it varies from one judgement to the next around an inferred average, with an inferred spread. So on some judgements a set that fits a rule is read as a range and the person endorses non-members inside the examples' span, and on others the same set is read as a rule. This blurs the commitment to rule or range most where the two are close competitors, such as clustered examples that share only a broad rule like "odd", and least where one explanation wins decisively.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment3/model_loop/models/variable_rule_range_expectation.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
