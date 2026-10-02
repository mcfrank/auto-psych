# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## personal_ideal_alternation  — rank 0, the best model on this data, ELPD-LOO -1008.9

Each person carries their own ideal switching rate for a random coin (how often consecutive flips should differ), and judges a sequence as more random the closer its proportion of alternations lies to that personal ideal; these ideals differ between people, some expecting near-perfect alternation and others streakier sequences. The model disagrees most with the current best model on highly alternating sequences (e.g. HTHTHTHT versus a streaky sequence), where it predicts a population split — some people strongly prefer them, others strongly reject them — and on pairs that differ in balance but not in alternation rate, where it predicts indifference.

## motif_stack_alternation_individual  — rank 1, 34.9 ± 19.3 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1043.8

Refinement of `motif_stack`: people judge randomness as the log-likelihood ratio of a fair coin versus Griffiths et al.'s four-motif stack automaton (the automaton held at the values the motif_stack fit settled on for this data), but people differ in how much they additionally favour or disfavour alternation — each participant carries their own preference for the sequence that switches between H and T more often, drawn from a population distribution. The single change is this participant-level alternation-preference random effect, added because the critique shows the between-participant spread in choosing the more-alternating sequence (SD 0.22) is four times what a single-population model produces.

## individual_alternation_prototype  — rank 2, 40.2 ± 11.2 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1049.1

Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular, i.e. close to a prototype alternation rate and not periodic). The one change: each person holds their own prototype alternation rate — some expect a random sequence to switch much more often than a fair coin does, others barely more or even less — drawn from a population distribution, instead of everyone sharing one prototype. This addresses the critique that people differ far more in how strongly they prefer the more-alternating sequence than a single-population model produces.

## individual_irregularity_weight  — rank 3, 71.0 ± 21.7 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1079.9

Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

## personal_switch_belief  — rank 4, 81.6 ± 21.8 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1090.6

Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

## local_representativeness  — rank 5, 389.5 ± 37.6 nats behind the best (10.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1398.4

A quantitative operationalization of Kahneman & Tversky (1972): local
balance is averaged across the whole sequence and sliding-window scales
2--4, while irregularity combines an over-alternating prototype with a
periodic-template penalty. This is theory-inspired because K&T did not
publish a unique quantitative scoring equation.

## motif_stack  — rank 6, 407.6 ± 40.6 nats behind the best (10.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1416.5

Randomness = log-likelihood ratio of a fair coin versus Griffiths et
al. (2018)'s four-motif stack automaton: a row-normalised six-state
motif process augmented with mirror symmetry, complement symmetry, and
duplication production methods, using the paper's max-path/max-method
definition. Comparisons are restricted to equal-length sequences.

## online_transition_surprise  — rank 7, 546.8 ± 53.0 nats behind the best (10.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1555.7

People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

## falk_konold_dp  — rank 8, 570.5 ± 44.4 nats behind the best (12.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1579.4

Sequences seem random to the extent they are hard to encode mentally:
randomness = the Difficulty Predictor DP = pure runs + 2*alternating
runs under the DP-minimising parse, unnormalised by length (Falk &
Konold 1997, p. 308). No free cognitive parameters.

## finite_experience_occurrence  — rank 9, 739.3 ± 68.8 nats behind the best (10.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.3

A string seems random to the extent one actually encounters it when
watching a fair coin for the paper's focal finite stretch: randomness =
log probability of occurring at least once in 20 flips (Hahn & Warren
2009). Comparisons are restricted to equal-length sequences. Penalises
long runs and perfect alternation without fitted cognitive parameters.
