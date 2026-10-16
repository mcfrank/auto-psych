# Candidate Brief

Refine the incumbent: `gamblers_second_order_person_lapse` — rank 0, the best model on this data, ELPD-LOO -3602.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `gamblers_run_second_order_side_habit`: people still judge a sequence random to the extent their own second-order, gambler's-fallacy model of a fair coin explains it better than the regular generators they suspect (a switch-biased coin, a biased coin, a repeating short motif with slips, with each person's own motif suspicion), the evidence weighed per flip by a fitted power of length and each person's side habit deciding near-ties. The single change is that people differ in how often they disengage: on a person-specific share of trials (drawn from a population, so a few participants are near-random responders while most rarely lapse) a participant does not judge the pair at all and picks Left or Right at random, which caps how consistently that person agrees with the majority even on clear-cut pairs — addressing the critique that participants differ in their rate of agreeing with the majority more than graded person-level sensitivity produces.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment3/model_loop/models/gamblers_second_order_person_lapse.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
