# Candidate Brief

Refine the incumbent: `graded_exemplar_concept_retrieval` — rank 0, the best model on this data, ELPD-LOO -2049.7. Its hypothesis, as its author stated it:

> People do not check whether a number literally belongs to a concept; they hold familiar number concepts (parity, primes, squares and cubes, powers, multiples, final digit, contained digit, and magnitude ranges of 1-100) in memory as stored lists of member numbers, and compare numbers to those stored members by closeness on the number line. A stored concept is retrieved to the extent that each example resembles its members on average (so small concepts whose members sit right at the examples win, and an example just next to a concept still partly supports it), with rules judged plausible only by their kind and an uncertain belief about how often an example is an unrelated stray; a target is then endorsed by how close it is to the nearest stored member of the retrieved concepts, so endorsement falls off gradedly just outside a concept instead of dropping to zero.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/graded_exemplar_concept_retrieval.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
