# Candidate Brief

Refine the incumbent: `range_restricted_rule_concepts` — rank 0, the best model on this data, ELPD-LOO -2109.2. Its hypothesis, as its author stated it:

> People do the same Bayesian concept learning as the best model (42 arithmetic rules and every interval of 1-100, a kind-level rule prior, an Erlang width prior favouring decade-bounded ranges, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their hypothesis space also contains range-restricted rules — conjunctions such as "even numbers in the 60s-70s" or "multiples of 4 below 20" — which receive an inferred share of prior belief and, being smaller, are strongly favoured by the size principle when the examples cluster. The sharpest disagreement with the best model lives on rule-consistent targets far from the examples' span (e.g. 12 or 30 after 66,78; 64 or 100 after 4,16,12): the best model says "yes" to every member of the rule wherever it lies, while this model predicts that "yes" to rule members falls off outside the region the examples occupy.

Its source is `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment2/model_loop/models/range_restricted_rule_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
