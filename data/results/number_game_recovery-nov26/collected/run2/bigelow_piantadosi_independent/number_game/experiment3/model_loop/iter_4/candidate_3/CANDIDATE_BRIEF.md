# Candidate Brief

Refine the incumbent: `effortful_computed_rule_search` — rank 0, the best model on this data, ELPD-LOO -3499.6. Its hypothesis, as its author stated it:

> Considering arithmetic rules that have to be computed (primes, squares and cubes, powers, multiples other than 5) takes deliberate mental effort, so people only search for them on some of their judgments. On the others they generalize from the explanations that are visible at a glance: stretches of the number line, odd or even, shared final or contained digits, and multiples of 5. On every judgment they weigh whichever concepts they have considered by the same Bayesian logic. This predicts that even when examples fit a computed rule well (e.g. 16, 8, 64 as powers of 2), a fixed share of answers follow the examples' range or digit pattern instead, giving fewer yeses to distant rule members and more yeses to non-members inside the span than a learner who always considers every rule.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/effortful_computed_rule_search.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
