# Candidate Brief

Refine the incumbent: `sequence_opens_bottom_ends_top` — rank 0, the best model on this data, ELPD-LOO -3098.0. Its hypothesis, as its author stated it:

> Refinement of `last_example_marks_range_edge`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe the program may deliberately end its demonstration on its range's top number, so a last-shown largest example blocks generalization above it. The one change is where the bottom edge is read from. It comes from the opening of the sequence, not its end: when several examples are shown and the smallest comes first, people believe (with separately inferred odds) that the program deliberately started at its range's bottom, and ranges whose bottom is that example get generative credit (larger for wider ranges). So after 20,35,60 people say "yes" less to 15 than after 35,20,60, which addresses the critique that a first-shown minimum restricts downward generalization.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/sequence_opens_bottom_ends_top.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
