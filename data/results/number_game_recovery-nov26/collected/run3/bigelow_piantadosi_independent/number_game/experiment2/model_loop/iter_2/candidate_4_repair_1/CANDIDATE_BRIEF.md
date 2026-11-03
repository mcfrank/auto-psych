# Candidate Brief

Refine the incumbent: `hundred_chart_row_ranges` — rank 0, the best model on this data, ELPD-LOO -2220.8. Its hypothesis, as its author stated it:

> Refinement of `factor_graded_multiples_concepts`: everything is kept (primitive-built rules with a fitted cost per extra primitive, a final-digit discount, extra costs for powers/cubes and for squares, times-table-graded multiples costs, Erlang-weighted ranges, the size principle and near-all-or-none commitment). The one change is how people picture numerical ranges: they lay 1-100 out as a hundred chart of ten rows (1-10, 11-20, ..., 91-100), so ranges made of whole rows ("1 to 50", "21 to 40", "81 to 90", "1 to 100") get a fitted prior bonus over ranges with arbitrary endpoints. When no rule explains the examples well, people therefore fall back on a broad round block of the chart and say yes fairly uniformly to targets within it, rather than to a tight range hugging the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment2/model_loop/models/hundred_chart_row_ranges.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
