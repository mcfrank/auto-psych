# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## lapse_individual_balance_alternation  — rank 0, the best model on this data, ELPD-LOO -952.4

Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.

## personal_lapse_ideal_alternation  — rank 1, 20.9 ± 8.9 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -973.3

People judge a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but the decision rule is not always engaged: on some trials a person does not compare the sequences at all and picks a side by guessing. How often this happens is a stable trait that differs between people, so some participants follow their alternation preference almost every trial while others are close to indifferent, which spreads individual choice proportions beyond what a single shared decisiveness produces.

## individual_balance_ideal_alternation  — rank 2, 37.0 ± 12.9 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -989.3

Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random, but people differ in how much that imbalance matters to them — each person has their own weight on H/T balance, drawn from a population distribution, instead of everyone sharing one. The one change is making the balance penalty individual, because the critique shows participants differ in their preference for the more balanced sequence far more than a single shared balance weight allows.

## ideal_alternation_with_balance  — rank 3, 56.1 ± 13.4 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1008.5

Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people additionally expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how it alternates. The one change is this shared penalty on H/T imbalance, added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which the alternation-only model predicts as indifference.

## windowed_glimpse_expectation  — rank 4, 56.2 ± 13.6 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1008.6

People read a coin-flip sequence through a limited working-memory window, holding only the last few flips at a time, and compare how mixed each glimpse is (its balance of heads and tails) with how mixed they expect a glimpse of a random coin to be; a sequence looks random to the extent its glimpses match that expectation. The expectation comes from each person's own belief about how often a random coin switches sides, so some people expect near-perfect mixing and others streakier glimpses, and because the window spans more than two flips, lopsided heads/tails counts and long runs are judged beyond what the alternation rate alone shows.

## personal_ideal_alternation  — rank 5, 56.5 ± 13.8 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1008.9

Each person carries their own ideal switching rate for a random coin (how often consecutive flips should differ), and judges a sequence as more random the closer its proportion of alternations lies to that personal ideal; these ideals differ between people, some expecting near-perfect alternation and others streakier sequences. The model disagrees most with the current best model on highly alternating sequences (e.g. HTHTHTHT versus a streaky sequence), where it predicts a population split — some people strongly prefer them, others strongly reject them — and on pairs that differ in balance but not in alternation rate, where it predicts indifference.

## ideal_alternation_streak_penalty  — rank 6, 57.0 ± 13.6 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1009.4

Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also notice the single longest streak of identical flips and treat a long streak as a sign of non-randomness, beyond what the overall alternation rate shows. The one change is a shared penalty on the length of the longest run (in flips), added because the critique shows that among irregular sequences people penalise long runs beyond what alternation rate captures, which the alternation-only model under-predicts.

## length_scaled_ideal_alternation  — rank 7, 57.2 ± 13.5 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1009.6

Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still look less random, but people treat a departure from their ideal switching rate as evidence that grows with the number of flips it is observed over — the same off-ideal alternation rate is shrugged off in a short sequence and taken seriously in a long one. The one change is making alternation sensitivity grow with sequence length (as a power of the number of transitions, fitted), because the critique shows short sequences are judged by alternation less, relative to long ones, than the length-independent incumbent predicts.

## iter1_candidate1  — rank 8, 61.0 ± 15.3 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1013.4

People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent it is better explained by a fair coin than by the non-random generators one might suspect — a biased coin (which explains unbalanced heads/tails counts) or a sticky/switchy coin (which explains long runs or rigid alternation). The one distortion is that each person's mental model of a "fair" coin switches between heads and tails at their own personal rate rather than exactly half the time, so people disagree about alternation while all still penalise imbalance and runs that the alternative generators explain.

## divisive_contrast_ideal_alternation  — rank 9, 63.3 ± 14.6 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1015.7

People do not judge each sequence's randomness on an absolute scale; they judge the two sequences against each other by divisive contrast: how much less random one looks than the other is weighed relative to how non-random the pair looks overall. Each sequence's non-randomness is its departure from the person's own ideal switching rate plus its heads/tails imbalance, but a given difference decides the choice strongly when the partner is nearly ideal and only weakly when both sequences are clearly non-random, so the same sequence is judged differently beside a different partner.

## motif_stack_alternation_individual  — rank 10, 91.4 ± 22.9 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1043.8

Refinement of `motif_stack`: people judge randomness as the log-likelihood ratio of a fair coin versus Griffiths et al.'s four-motif stack automaton (the automaton held at the values the motif_stack fit settled on for this data), but people differ in how much they additionally favour or disfavour alternation — each participant carries their own preference for the sequence that switches between H and T more often, drawn from a population distribution. The single change is this participant-level alternation-preference random effect, added because the critique shows the between-participant spread in choosing the more-alternating sequence (SD 0.22) is four times what a single-population model produces.

## individual_alternation_prototype  — rank 11, 96.7 ± 18.7 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1049.1

Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular, i.e. close to a prototype alternation rate and not periodic). The one change: each person holds their own prototype alternation rate — some expect a random sequence to switch much more often than a fair coin does, others barely more or even less — drawn from a population distribution, instead of everyone sharing one prototype. This addresses the critique that people differ far more in how strongly they prefer the more-alternating sequence than a single-population model produces.

## max_lopsided_stretch_ideal  — rank 12, 97.6 ± 26.2 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1050.0

People judge a coin-flip sequence by its single most lopsided stretch: they notice the one contiguous run of flips in which heads most outnumber tails (or tails most outnumber heads), and a sequence looks random to the extent that this worst local excess matches what each person expects a random coin to produce. Each person carries their own ideal size for this worst stretch (relative to the sequence's length) — some expect even the most uneven stretch to be tiny, others tolerate a sizeable one — and the rest of the sequence, such as its overall alternation rate, matters only through that one most striking stretch.

## local_window_ideal_alternation_lapse  — rank 13, 98.8 ± 16.3 nats behind the best (6.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1051.1

Refinement of the incumbent `personal_lapse_ideal_alternation`: people still judge a sequence as more random the closer its switching between heads and tails lies to their own personal ideal switching rate, and still guess on some trials at a personal, trait-like lapse rate, but they apply their ideal locally rather than to the sequence as a whole — they read the sequence a few flips at a time (a sliding stretch of three consecutive transitions) and each stretch is expected to switch at about the ideal rate, so a sequence's non-randomness is its average local departure from the ideal. The one change is this local (windowed) application of the ideal instead of the global alternation rate: two sequences with the same overall alternation rate differ when one bunches its repeats into a long streak and its switches into a rigid alternating stretch, which is exactly the streak aversion among alternation-matched pairs that the critique shows the incumbent under-predicts.

## run_hazard_switch_belief  — rank 14, 100.2 ± 22.6 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1052.6

Refinement of `personal_switch_belief` (each person judges as more random the sequence that is more probable under their own subjective model of a coin as a process that switches between heads and tails with a personal probability). The one change: the believed coin has the gambler's fallacy built in — its chance of switching grows with the length of the current run (a shared rate of growth), on top of each person's own baseline switch belief — so a long run is judged far less probable than its number of switches alone implies. This addresses the critique that people penalise long runs beyond what the alternation rate captures.

## personal_ideal_longest_run  — rank 15, 112.1 ± 28.3 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1064.5

People judge randomness by the longest streak a sequence contains: each person carries their own ideal length for the longest run of identical flips in a random sequence (relative to its length), and picks the sequence whose longest streak is closer to that personal ideal. Some people expect random sequences to contain almost no streaks, others expect a noticeable streak, so the same pair can be judged in opposite directions; the rest of the sequence's structure (its overall alternation rate or H/T balance) plays no role beyond what it does to the longest streak.

## individual_irregularity_weight  — rank 16, 127.5 ± 24.6 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1079.9

Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

## personal_switch_belief  — rank 17, 138.2 ± 25.8 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1090.6

Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

## tally_drift_ideal  — rank 18, 164.9 ± 32.5 nats behind the best (5.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1117.3

People keep a running tally of heads minus tails as they read a sequence, and judge it random to the extent that this tally wanders away from balance by about as much as they expect a fair coin's tally to wander: the whole path matters, so a streak that pushes the tally far off balance (even one later corrected) counts against a sequence, and so does a tally that never leaves balance at all. Each person has their own ideal amount of wandering — some expect the tally to hug zero, others tolerate large drifts — and picks the sequence whose tally drift is closer to that personal ideal.

## pattern_coverage_detection  — rank 19, 173.8 ± 29.1 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1126.1

People judge randomness by pattern detection: they scan a sequence for perceptible regularities — a streak of identical flips or a stretch of strict H/T alternation — and a sequence looks non-random in proportion to how much of it is covered by such a detected pattern. A stretch is noticed as a pattern once it is long enough, with streaks noticed at a shared length and alternating stretches at a length that differs from person to person, so the whole shape of the sequence (how many of its flips sit inside long runs or long alternations, not its overall switch rate) drives the choice.

## exemplar_designed_pattern_similarity  — rank 20, 266.9 ± 27.0 nats behind the best (9.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1219.3

People judge randomness by exemplar similarity: they carry a few remembered examples of obviously designed coin sequences — a solid streak (HHHH…), strict alternation (HTHT…), pairs (HHTT…), triples (HHHTTT…) and two halves (HHHH TTTT) — and a sequence looks non-random to the extent that it closely resembles (differs in few flips from) any of these remembered patterns, with similarity falling off exponentially with the share of mismatched flips. People differ in whether their remembered set of "designed" sequences prominently includes strict alternation, so some find alternating sequences contrived and others find them random-looking, while streak-like resemblance is penalised by everyone.

## local_representativeness  — rank 21, 446.0 ± 39.5 nats behind the best (11.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1398.4

A quantitative operationalization of Kahneman & Tversky (1972): local
balance is averaged across the whole sequence and sliding-window scales
2--4, while irregularity combines an over-alternating prototype with a
periodic-template penalty. This is theory-inspired because K&T did not
publish a unique quantitative scoring equation.

## motif_stack  — rank 22, 464.1 ± 42.7 nats behind the best (10.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1416.5

Randomness = log-likelihood ratio of a fair coin versus Griffiths et
al. (2018)'s four-motif stack automaton: a row-normalised six-state
motif process augmented with mirror symmetry, complement symmetry, and
duplication production methods, using the paper's max-path/max-method
definition. Comparisons are restricted to equal-length sequences.

## online_transition_surprise  — rank 23, 603.4 ± 52.9 nats behind the best (11.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1555.7

People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

## falk_konold_dp  — rank 24, 627.0 ± 46.4 nats behind the best (13.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1579.4

Sequences seem random to the extent they are hard to encode mentally:
randomness = the Difficulty Predictor DP = pure runs + 2*alternating
runs under the DP-minimising parse, unnormalised by length (Falk &
Konold 1997, p. 308). No free cognitive parameters.

## finite_experience_occurrence  — rank 25, 795.9 ± 68.1 nats behind the best (11.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1748.3

A string seems random to the extent one actually encounters it when
watching a fair coin for the paper's focal finite stretch: randomness =
log probability of occurring at least once in 20 flips (Hahn & Warren
2009). Comparisons are restricted to equal-length sequences. Penalises
long runs and perfect alternation without fitted cognitive parameters.
