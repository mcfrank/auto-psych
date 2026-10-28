# Candidate Brief

Refine the incumbent: `compressed_rule_extension` — rank 0, the best model on this data, ELPD-LOO -3098.2. Its hypothesis, as its author stated it:

> People think of an arithmetic property such as "even numbers" or "multiples of 5" by its description rather than by listing its members, so they underestimate how many numbers it contains. When they weigh how well a concept explains the examples (the size principle), they treat a rule as smaller than it really is by an inferred degree, but they judge a numeric range at its true size, because its extent is visible on the number line. Otherwise they reason as in Bayesian concept learning over rules, ranges and range-restricted rules, with the last-shown largest example marking a range's top. So broad properties like parity keep more support against tight ranges, and people generalize by the shared property beyond the examples' span (yes to same-parity numbers far away, no to opposite-parity neighbours just outside) more than an ideal learner would.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/compressed_rule_extension.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
