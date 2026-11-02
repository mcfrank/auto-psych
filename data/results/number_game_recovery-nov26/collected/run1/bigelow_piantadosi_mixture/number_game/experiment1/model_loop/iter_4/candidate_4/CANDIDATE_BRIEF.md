# Candidate Brief

Refine the incumbent: `sticky_guess_power_law_ranges` — rank 0, the best model on this data, ELPD-LOO -866.1. Its hypothesis, as its author stated it:

> Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, as later examples arrive as long as each new example fits it, reconsidering from all examples so far only when it is contradicted (or they do not stick). The one change, the component taken from power_law_interval_endpoints: when they consider a numerical range they do not treat all locations on the number line alike — the range's endpoints have a prior plausibility that follows a fitted power law over 1-100 (on top of the incumbent's Erlang preference for a typical width), so whether a range guess is formed and kept depends on whether the examples are small or large numbers.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/sticky_guess_power_law_ranges.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
