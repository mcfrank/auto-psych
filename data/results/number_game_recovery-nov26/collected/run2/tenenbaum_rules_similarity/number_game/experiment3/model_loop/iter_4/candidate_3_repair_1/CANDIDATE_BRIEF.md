# Candidate Brief

Refine the incumbent: `new_set_criterion_ratio_ceiling` — rank 0, the best model on this data, ELPD-LOO -2006.5. Its hypothesis, as its author stated it:

> I refined the incumbent graded_ratio_ceiling_criterion, grafting the new-set caution of the pruned new_set_cautious_erlang_concepts. People do the same Bayesian concept learning (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement, no "contains digit d" concepts) and treat the largest example as a soft, ratio-graded ceiling; the one change is that their yes/no criterion also depends on whether the set is new: on the first question after the examples change (or at session start) they demand a different amount of certainty to say "yes" than on a repeated question about the same set, a single log-odds criterion shift whose size and sign are inferred (zero recovers the incumbent). This predicts that, at equal belief, the first query about a new set gets fewer "yes" answers than later queries about that same set, as the posterior-predictive critique of the incumbent suggests.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment3/model_loop/models/new_set_criterion_ratio_ceiling.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
