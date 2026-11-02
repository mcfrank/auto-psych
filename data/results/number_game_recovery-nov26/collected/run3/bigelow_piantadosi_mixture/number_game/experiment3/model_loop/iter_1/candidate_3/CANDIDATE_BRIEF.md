# Candidate Brief

Refine the incumbent: `range_limited_rule_familiar_strays` — rank 0, the best model on this data, ELPD-LOO -3325.3. Its hypothesis, as its author stated it:

> Refinement of `rule_count_familiar_strays`: people do the same Bayesian concept learning over equally plausible mathematical rules and magnitude ranges of 1-100, with the size principle, an uncertain belief about how often an example is a stray, and strays that are sometimes familiar numbers (familiarity = how many rules a number belongs to). The one change is the concept space: people also consider that the program applies a rule only within a magnitude range (e.g. "odd numbers between 9 and 33", "multiples of 4 from 52 to 88"), with an inferred share of their belief in rules going to such range-limited rules. So when the examples fit a rule and sit in part of the number line, people endorse the rule's members inside the examples' span more than its members far outside it, instead of treating every member of the rule alike.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment3/model_loop/models/range_limited_rule_familiar_strays.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
