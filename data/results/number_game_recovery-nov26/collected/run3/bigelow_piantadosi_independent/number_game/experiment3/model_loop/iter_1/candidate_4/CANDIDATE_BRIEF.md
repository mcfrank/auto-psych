# Candidate Brief

Refine the incumbent: `fives_privileged_decoupled_prime_odd` — rank 0, the best model on this data, ELPD-LOO -3549.1. Its hypothesis, as its author stated it:

> Refinement of `prime_bonus_odd_cost_decoupled`, with everything in it kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, a separate fitted bonus for "prime numbers" and cost for "odd numbers", the repeated-digit rule, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change is that "multiples of 5" is unlike the other times tables. Counting by fives is a drilled skill, and membership can be read straight off the numeral (it ends in 0 or 5), so this one rule gets its own fitted prior bonus. So after sets of multiples of 5 that are not all multiples of 10 (e.g. 90, 100, 5 or 35, 15, 85), people commit to "multiples of 5": they say yes to other numbers ending in 0 or 5 and no to neighbouring non-multiples more sharply than the base model predicts, while sets that fit other times tables are unaffected.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/fives_privileged_decoupled_prime_odd.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
