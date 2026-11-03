# Candidate Brief

Refine the incumbent: `digit_rule_discount_sticky_guess` — rank 0, the best model on this data, ELPD-LOO -2100.4. Its hypothesis, as its author stated it:

> Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted, with range guesses shaped by a power-law endpoint and Erlang width prior. The one change: rules defined only by how a number is written ("ends in 7", "contains the digit 3") are a priori less (or more) plausible than arithmetic rules (odd, multiples, powers, squares, primes), by a single fitted weight on the digit-rule family — so people latch onto and extend "same last digit" guesses to far-away numbers less readily than the incumbent predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/digit_rule_discount_sticky_guess.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
