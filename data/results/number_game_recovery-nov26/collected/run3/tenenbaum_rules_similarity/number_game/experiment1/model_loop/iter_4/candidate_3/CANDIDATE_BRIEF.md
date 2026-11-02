# Candidate Brief

Refine the incumbent: `parity_within_range_concepts` — rank 0, the best model on this data, ELPD-LOO -1072.9. Its hypothesis, as its author stated it:

> Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/parity_within_range_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
