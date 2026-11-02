# Candidate Brief

Refine the incumbent: `erlang_width_compressed_intervals` — rank 0, the best model on this data, ELPD-LOO -1895.8. Its hypothesis, as its author stated it:

> Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, with interval members pictured as sampled along a compressed number line, but they do not consider every numerical range equally plausible beforehand — they expect a program that produces "a range of numbers" to produce a range of moderate width, so tiny ranges hugging one or two examples are a priori unlikely (an Erlang-shaped prior on range width, its typical width fitted). The one change is this width prior on intervals, replacing the flat prior: people fill in the whole span between the examples more, do not build a tight range around a lone exceptional example, and lean on parity or digit rules rather than a tiny range for a target just one step from an example.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/erlang_width_compressed_intervals.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
