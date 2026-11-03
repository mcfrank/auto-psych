# Candidate Brief

Refine the incumbent: `rule_count_familiar_strays` — rank 0, the best model on this data, ELPD-LOO -3325.9. Its hypothesis, as its author stated it:

> Refinement of `familiar_strays_equal_rule_prior`: people do the same Bayesian concept learning over equally plausible mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes "familiar" numbers rather than random ones. The one change is what makes a number familiar. It is not how readily a random member of a random rule lands on it, which favours members of tiny rules such as 27, 64 and 81. It is how many well-known rules the number belongs to, its breadth of mathematical associations. So richly connected numbers (12, 24, 36, 60, 100) are the easiest examples to write off as strays, while a lone member of a tiny rule is no more excusable than other numbers with as many associations. This adds no free parameter.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment3/model_loop/models/rule_count_familiar_strays.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
