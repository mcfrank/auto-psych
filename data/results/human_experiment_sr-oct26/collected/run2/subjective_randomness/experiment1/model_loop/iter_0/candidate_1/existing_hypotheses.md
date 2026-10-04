# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## local_representativeness  — rank 0, the best model on this data, ELPD-LOO -1380.4

A quantitative operationalization of Kahneman & Tversky (1972): local
balance is averaged across the whole sequence and sliding-window scales
2--4, while irregularity combines an over-alternating prototype with a
periodic-template penalty. This is theory-inspired because K&T did not
publish a unique quantitative scoring equation.

## motif_stack  — rank 1, 47.0 ± 21.6 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1427.4

Randomness = log-likelihood ratio of a fair coin versus Griffiths et
al. (2018)'s four-motif stack automaton: a row-normalised six-state
motif process augmented with mirror symmetry, complement symmetry, and
duplication production methods, using the paper's max-path/max-method
definition. Comparisons are restricted to equal-length sequences.

## falk_konold_dp  — rank 2, 184.1 ± 28.5 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1564.5

Sequences seem random to the extent they are hard to encode mentally:
randomness = the Difficulty Predictor DP = pure runs + 2*alternating
runs under the DP-minimising parse, unnormalised by length (Falk &
Konold 1997, p. 308). No free cognitive parameters.

## finite_experience_occurrence  — rank 3, 322.8 ± 47.6 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1703.2

A string seems random to the extent one actually encounters it when
watching a fair coin for the paper's focal finite stretch: randomness =
log probability of occurring at least once in 20 flips (Hahn & Warren
2009). Comparisons are restricted to equal-length sequences. Penalises
long runs and perfect alternation without fitted cognitive parameters.
