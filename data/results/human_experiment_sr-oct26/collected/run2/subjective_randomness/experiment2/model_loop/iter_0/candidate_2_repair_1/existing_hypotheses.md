# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## heads_default_alternation_ideal  — rank 0, the best model on this data, ELPD-LOO -2390.1

People do not treat the two faces of the coin symmetrically: heads is the default, expected outcome of a coin toss, so a sequence dominated by tails reads as a coin that is "off" (biased toward the unusual face) and looks less random, while a heads-leaning sequence looks like ordinary coin flipping. This label asymmetry operates on top of each person's judgement of how close a sequence's switching rate is to their own ideal (with person-specific, length-scaled sensitivity and a shared penalty for visibly periodic sequences): of two sequences that switch equally often, people pick the one with more heads.

## heads_favoring_lapse_ideal_streak  — rank 1, 28.4 ± 14.5 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -2418.5

Refinement of `personal_lapse_ideal_streak_excess`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, still penalises every flip of the longest streak beyond two, and still lapses to a random pick at their own rate — but the two coin faces are not psychologically symmetric: heads is the canonical, expected outcome of a coin toss, so a sequence with a larger share of heads reads as more like "real" coin flipping and looks more random. The one change is a shared heads-share bias inside the engaged decision, addressing the critique that among pairs with equal switch counts people choose the heads-heavier sequence more often than an H/T-symmetric model allows.

## iter0_candidate0  — no comparison row

People judge randomness by Bayesian model comparison: a sequence looks random to the extent that it is better explained by a random coin than by a "rigged" process — either a coin biased toward one face (of unknown bias) or a sticky/switchy coin that tends to repeat or alternate (of unknown tendency) — so lopsided H/T counts and extreme switching both count as evidence of a non-random generator. The one distortion is in each person's picture of the random coin itself: rather than a fair, memoryless coin, each person believes a random coin switches sides at their own personal rate (usually more than half the time), and they choose the sequence with the higher posterior odds of having come from that subjective random coin.

## gist_typicality_count_runs  — no comparison row

People judge how random a sequence looks by how typical its gist is for a fair coin: they register only two coarse summaries of a sequence — how many heads it has and how many runs (streaks) it breaks into — and a sequence looks random to the extent that many coin sequences of that length share that same gist. Lopsided counts, long streaks (few runs) and strict alternation (maximal runs) are all rare gists and look non-random, while balanced sequences with a middling number of runs are common gists and look random; people differ only in how strongly this felt typicality drives their choice.
