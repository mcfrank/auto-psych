# Candidate Brief

Refine the incumbent: `erlang_interval_prior_concepts` — rank 0, the best model on this data, ELPD-LOO -869.6. Its hypothesis, as its author stated it:

> Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule), but they do not regard every interval as equally plausible a priori. Instead, their prior over intervals follows an Erlang density on the interval's size with a fitted characteristic size, so very narrow ranges hugging the examples are a priori implausible and ranges of a typical intermediate width are favoured; this one change controls how far beyond the examples' span people generalise when they think in terms of a numerical range.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_prior_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
