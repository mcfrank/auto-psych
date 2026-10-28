# Candidate Brief

Refine the incumbent: `prime_concept_excludes_one` — rank 0, the best model on this data, ELPD-LOO -2060.6. Its hypothesis, as its author stated it:

> Refinement of `range_restricted_rule_concepts`: people do the same Bayesian concept learning over arithmetic rules, every range of 1-100 and range-restricted rules (kind-level rule prior, Erlang width prior with a decade-boundary preference, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their concept of "prime numbers" is the mathematical one, which excludes 1. The one change is removing 1 from the prime concept: when 1 is among the examples (e.g. 31,3,1) people cannot explain the set as primes and fall back on "odd numbers" (yes to 9 or 25, no special yes to 2), and after a set of primes (e.g. 67,31,17,61) they answer "no" to 1 rather than accepting it as a prime.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/prime_concept_excludes_one.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
