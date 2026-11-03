# Candidate Brief

Refine the incumbent: `rule_plus_exception_neighbourhood` — rank 0, the best model on this data, ELPD-LOO -1760.3. Its hypothesis, as its author stated it:

> Refinement of `rule_plus_exception_concepts`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus exception" concepts, but the exception they add to a rule is not a single isolated number — it is a small fuzzy region of the number line around it ("ends in 2, and also numbers around 27"), whose radius (zero, one, two … up to three steps) is a priori more likely the smaller it is, with the fall-off fitted. The one change is this neighbourhood-sized exception: when one example breaks the others' pattern (for instance, the one odd number among even ones), people still endorse it and still extend the rule to distant numbers, but they also endorse the off-pattern example's close neighbours more than the point-exception model predicts, as the critique reports.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/rule_plus_exception_neighbourhood.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
