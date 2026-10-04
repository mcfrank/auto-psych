# Candidate Brief

Refine the incumbent: `terminal_streak_recency_pattern_rule` — rank 0, the best model on this data, ELPD-LOO -3262.5. Its hypothesis, as its author stated it:

> Refinement of the incumbent `person_switch_ideal_pattern_rule_built` (tally span judged against a personal expected span, switching judged by closeness to a personal ideal switching rate, chunk variety and a rule-built penalty scaled by a personal pattern sensitivity, a personal left/right lean). The one change: people weigh the end of a sequence, the flips they read last, more than its interior — a sequence that finishes on a streak (the coin "getting stuck" at the end, HTHHTT, THTHHH) looks less random, and every flip of the final run beyond the first counts against it, by a shared terminal-streak penalty. Of two sequences with the same number of switches, people therefore pick the one that ends with a switch, addressing the critique that among equal-switch pairs people choose the sequence ending with a switch more often than the incumbent, which is blind to where switches occur, predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/terminal_streak_recency_pattern_rule.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
