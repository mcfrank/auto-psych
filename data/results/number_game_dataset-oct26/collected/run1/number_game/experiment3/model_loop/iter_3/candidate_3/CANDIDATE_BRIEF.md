# Candidate Brief

Refine the incumbent: `privileged_named_number_concepts` — rank 0, the best model on this data, ELPD-LOO -4696.2. Its hypothesis, as its author stated it:

> A refinement of `even_numbers_privileged_rule` (shortfall from the examples' mutual final-digit similarity, rule-pre-empted parity credit, rule-gated decade-band familiarity, and the privileged "even numbers" rule that makes all-even sets reject odd targets extra strongly, all kept). The one change, a component taken from `round_counting_privileged_factor`, is that counting by fives and tens is also a privileged, overlearned concept: when the examples' common factor is a multiple of 5 people embrace "multiples of that factor" with its own, inferred strength, separate from the boost for non-round factors such as 3, 4, 6 or 8. So people favour the culturally named number concepts — evens, and counting by fives and tens — over equally consistent but unnamed ones (odd numbers, multiples of 3, 4 or 8).

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/privileged_named_number_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
