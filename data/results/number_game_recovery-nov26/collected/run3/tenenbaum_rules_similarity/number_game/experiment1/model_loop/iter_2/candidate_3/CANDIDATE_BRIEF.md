# Candidate Brief

Refine the incumbent: `erlang_interval_family_prior_concepts` — rank 0, the best model on this data, ELPD-LOO -1083.6. Its hypothesis, as its author stated it:

> Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_family_prior_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
