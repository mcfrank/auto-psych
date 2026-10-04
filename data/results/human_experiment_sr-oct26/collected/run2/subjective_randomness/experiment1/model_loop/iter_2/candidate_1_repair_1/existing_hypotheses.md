# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## periodic_penalized_alternation_ideal  — rank 0, the best model on this data, ELPD-LOO -1072.0

Refinement of the incumbent `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but in addition everyone notices when a sequence is built by repeating a short unit (e.g. HTHTHTHT, HHTTHHTT, HTTHTTHT) and counts that visible periodic pattern against its randomness. The one change is this shared periodic-pattern penalty, addressing the critique that people reject regular repeating patterns (and perfect alternation) far more than the alternation-distance mechanism alone predicts.

## streak_penalized_personal_prototype  — rank 1, 6.6 ± 4.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1078.6

Refinement of `person_specific_alternation_prototype` (local representativeness with a person-specific preferred alternation rate): people judge randomness by multiscale local H/T balance and by irregularity — how far a sequence's switching rate is from their own preferred rate, plus a periodic-template penalty — and, as the one added component, they treat the longest streak in a sequence as a salient sign of non-randomness, penalising long runs beyond what the overall switching rate implies. (The distance from the personal preferred rate is taken as smooth and squared, as in the incumbent, rather than absolute; the claim is the added streak penalty.) This targets the critique that the best model under-penalises long streaks and regular patterns once alternation rate is accounted for.

## personal_alternation_ideal_streak_penalty  — rank 2, 12.2 ± 6.4 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1084.2

Refinement of `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but people also treat the longest streak of identical outcomes as a salient sign of non-randomness in its own right. The one change is a shared penalty on the sequence's longest run (relative to its length), so that of two sequences equally close to a person's ideal switching rate, the one containing a longer streak looks less random — addressing the critique that long streaks are penalised beyond what the alternation rate explains.

## personal_alternation_ideal  — rank 3, 16.2 ± 8.1 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1088.2

Each person carries their own ideal switching rate for a random coin — some expect a fair coin to flip side about half the time, others expect it to alternate far more often — and judges a sequence as random to the extent that its proportion of H/T switches is close to that personal ideal. The current best model assumes one shared alternation prototype for everyone; this model disagrees most sharply on pairs where both sequences alternate heavily (e.g. HTHTHTHT versus HTHHTHTH), which strong over-alternators should split decisively toward the more alternating sequence while moderate people choose the other.

## person_specific_alternation_prototype  — rank 4, 55.7 ± 14.4 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1127.7

Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge randomness by local H/T balance across scales plus irregularity measured as distance from a preferred alternation rate and a periodic-template penalty, but each person holds their own preferred alternation rate (prototype) rather than one shared by everyone. The single change is making the alternation prototype person-specific, drawn from a population distribution, which addresses the critique that the model under-produces individual differences in preference for alternation.

## motif_stack_person_alternation  — rank 5, 89.2 ± 29.4 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1161.2

Refinement of `motif_stack`: people judge randomness as the likelihood ratio of a fair coin against Griffiths et al.'s four-motif stack automaton, a regularity detector shared by everyone (held at the automaton settings `motif_stack` itself estimated on these data), but individuals differ in how strongly they additionally treat frequent alternation itself as a sign of randomness. The one change is a person-specific alternation-preference weight, drawn from a population distribution, on the difference in alternation rate between the two sequences — addressing the critique that people differ far more in their preference for alternation than a single-population model produces.

## person_alternation_prototype  — rank 6, 89.5 ± 18.5 nats behind the best (4.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1161.5

Refinement of the incumbent `local_representativeness` (Kahneman & Tversky local representativeness: a sequence looks random when it is locally balanced and irregular, irregularity being closeness to an over-alternating prototype plus a periodic-template penalty). The one change: each person carries their own ideal alternation rate (prototype), drawn from a population distribution, rather than everyone sharing a single prototype — so people differ in how much alternation they expect from a random coin, which the single-population incumbent cannot produce (the critique's large across-participant spread in preference for the more-alternating sequence).

## bayes_pattern_vs_alternating_coin  — rank 7, 127.7 ± 23.6 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1199.7

People judge randomness by Bayesian model comparison: a sequence looks random to the extent it is better explained by a random coin than by a noisy repeating pattern (a short template of period 1-4, such as HHHH, HTHT or HHTT, copied with occasional errors). The single distortion is in their picture of the random coin: each person believes a fair coin switches sides at their own rate (often more than half the time), so alternation counts as evidence for randomness up to the point where the sequence becomes regular enough to be better explained by a repeating template.

## personal_markov_coin_belief  — rank 8, 179.9 ± 33.2 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1251.9

Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

## position_weighted_switch_impression  — rank 9, 286.4 ± 43.4 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1358.4

People do not weigh every part of a sequence equally: they form their impression of how often the coin switches mainly from one end of the sequence (the last flips they read, or the first), and judge a sequence as more random the closer that position-weighted switch rate comes to a shared ideal switch rate for a random coin. Two sequences with the same overall number of switches can therefore be judged differently depending on whether their switches or their repeats sit at the attended end.

## local_representativeness  — rank 10, 308.4 ± 41.6 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1380.4

A quantitative operationalization of Kahneman & Tversky (1972): local
balance is averaged across the whole sequence and sliding-window scales
2--4, while irregularity combines an over-alternating prototype with a
periodic-template penalty. This is theory-inspired because K&T did not
publish a unique quantitative scoring equation.

## motif_stack  — rank 11, 355.4 ± 46.2 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1427.4

Randomness = log-likelihood ratio of a fair coin versus Griffiths et
al. (2018)'s four-motif stack automaton: a row-normalised six-state
motif process augmented with mirror symmetry, complement symmetry, and
duplication production methods, using the paper's max-path/max-method
definition. Comparisons are restricted to equal-length sequences.

## context_learner_surprise  — rank 12, 487.0 ± 64.2 nats behind the best (7.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1559.0

People read a sequence flip by flip and, with a short memory of the last two flips, keep trying to predict the next flip from what followed the same two-flip context earlier in the sequence (an online pattern learner); a sequence looks random to the extent that these running predictions keep failing, i.e. by the average surprise the learner experiences. Long streaks and repeating patterns (HTHTHTHT, HHTTHHTT) quickly become predictable to such a learner and so look non-random even when their overall switch rate is near what people expect, and people differ in how strongly this felt surprise drives their choice.

## falk_konold_dp  — rank 13, 492.5 ± 44.7 nats behind the best (11.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1564.5

Sequences seem random to the extent they are hard to encode mentally:
randomness = the Difficulty Predictor DP = pure runs + 2*alternating
runs under the DP-minimising parse, unnormalised by length (Falk &
Konold 1997, p. 308). No free cognitive parameters.

## best_lag_copy_rule_detector  — rank 14, 584.7 ± 64.0 nats behind the best (9.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.7

People judge randomness by searching each sequence for a simple copy rule — "each flip repeats the flip k places back" for some small lag k (lag 1 catches streaks, lag 2 catches HTHT alternation, lag 4 catches HHTTHHTT) — and a sequence looks non-random to the extent that its best such rule predicts its flips. They choose the sequence whose best copy rule fits worse, with people differing only in how strongly that detected regularity drives their choice.

## finite_experience_occurrence  — rank 15, 631.2 ± 67.1 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1703.2

A string seems random to the extent one actually encounters it when
watching a fair coin for the paper's focal finite stretch: randomness =
log probability of occurring at least once in 20 flips (Hahn & Warren
2009). Comparisons are restricted to equal-length sequences. Penalises
long runs and perfect alternation without fitted cognitive parameters.

## pair_normalized_alternation_contrast  — no comparison row

People do not judge each sequence's randomness on an absolute scale; they judge the pair relative to itself. Each person compares how far each sequence's switching rate is from their own ideal switching rate, but the difference between the two is weighed relative to how far both sequences are from that ideal together (divisive contrast normalisation, as in Weber's law): a given gap decides the choice sharply when both sequences are near the ideal and barely matters when both are far from it, so the same sequence is chosen more or less decisively depending on its partner.
