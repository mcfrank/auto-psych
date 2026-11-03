# Candidate Brief

Refine the incumbent: `repdigit_pattern_chart_concepts` — rank 0, the best model on this data, ELPD-LOO -3586.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `prime_over_odd_reading`, with everything in it kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, the single preference that moves plausibility from "odd" to "prime", Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change adds a concept the incumbent lacks. People notice when a numeral's two digits are the same (11, 22, ..., 99), so "repeated-digit numbers" is a candidate rule, priced like other two-primitive rules plus its own fitted prior bonus. After sets holding two or more repdigits (e.g. 33, 77 or 44, 11, 88), people say yes to other repdigits and no to numbers that merely lie near the examples, which closes the critique's report that the incumbent under-predicts yes to repdigit targets after repdigit sets.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/repdigit_pattern_chart_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
