# Candidate Brief

Refine the incumbent: `noisy_examples_erlang_intervals` — rank 0, the best model on this data, ELPD-LOO -1210.3. Its hypothesis, as its author stated it:

> A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate at which they believe examples are random noise, but they do not treat every interval as equally likely a priori. The one change is grafting the Erlang interval-size prior from erlang_interval_size_prior: people expect a "range" concept to have a typical width (inferred from the data), so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100, which governs how far they extend "yes" around examples that fit no rule.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/noisy_examples_erlang_intervals.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
