# Candidate Brief

Refine the incumbent: `sticky_guess_fitted_sampling_trust` — rank 0, the best model on this data, ELPD-LOO -2100.0. Its hypothesis, as its author stated it:

> Refinement of digit_rule_discount_sticky_guess: people hold a single working guess about the concept (a rule or a numerical range, with a discounted prior on digit-based rules and power-law/Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted. The one change (the component taken from inferred_sampling_reliability_concepts): how much people trust that the examples were really drawn from the program's concept, rather than being stray numbers, is fitted instead of fixed at 90% — so the strength of the size principle, and hence how readily people entertain (and answer by) guesses that do not even contain every example, is set by the data.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/sticky_guess_fitted_sampling_trust.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
