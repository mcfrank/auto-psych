# Candidate Brief

Refine the incumbent: `rule_plus_exception_concepts` — rank 0, the best model on this data, ELPD-LOO -1768.9. Its hypothesis, as its author stated it:

> Refinement of `final_digit_salient_rules`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, but their concept vocabulary also contains "rule plus one exception" concepts — a rule together with a single extra number it does not contain ("ends in 2, and also 27"), like the classic "powers of 2, or 37" — which share a fitted, reduced part of each rule's prior. The one change is these explicit rule-with-exception concepts: when one example breaks the others' pattern, people treat that example as a genuine member of the program's output rather than a random slip, so they endorse the off-pattern example itself almost unanimously while still rejecting its numerical neighbours and extending the rule to distant numbers.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/rule_plus_exception_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
