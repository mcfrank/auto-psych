# Candidate Brief

Refine the incumbent: `last_example_marks_range_top` — rank 0, the best model on this data, ELPD-LOO -3099.1. Its hypothesis, as its author stated it:

> Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random — but they believe the program ends its demonstration on that top number. The one change: the marked top is read from the order of presentation, so a range ending at the largest example gets the generative top-marking credit (larger for wider ranges) only when the largest example is the last one shown, and then at full strength whatever the set size, while sets whose largest example comes earlier get no edge credit — after 20,35,60 people say "yes" to 45 but not 62, whereas after 60,35,20 they extend above 60 as readily as below 20.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/last_example_marks_range_top.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
