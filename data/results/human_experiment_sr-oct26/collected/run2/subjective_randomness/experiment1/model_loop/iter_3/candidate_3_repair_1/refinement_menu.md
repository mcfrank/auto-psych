# Refinement menu

The models you may refine, other than the incumbent `length_scaled_alternation_ideal`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### personal_ideal_with_personal_lapse — rank 1, 1.3 ± 11.3 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -1071.2

**Hypothesis:** People judge which sequence looks more random by how close its proportion of H/T switches is to their own personal ideal switching rate, but the decision rule is not always engaged: each person has their own lapse rate, the share of trials on which they do not evaluate the sequences at all and pick a side at random. People therefore differ not only in what they think randomness looks like but in how consistently they act on it, so even a pair with a clear winner is split by some participants, more for some people than for others.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_ideal_with_personal_lapse.py`

### periodic_penalized_alternation_ideal — rank 2, 2.1 ± 2.6 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1072.0

**Hypothesis:** Refinement of the incumbent `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but in addition everyone notices when a sequence is built by repeating a short unit (e.g. HTHTHTHT, HHTTHHTT, HTTHTTHT) and counts that visible periodic pattern against its randomness. The one change is this shared periodic-pattern penalty, addressing the critique that people reject regular repeating patterns (and perfect alternation) far more than the alternation-distance mechanism alone predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/periodic_penalized_alternation_ideal.py`

### periodic_ideal_absolute_streak_penalty — rank 3, 2.4 ± 2.6 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -1072.3

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, with a shared penalty for visibly periodic sequences, but people also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness, however long the sequence. The one change is this shared penalty on the absolute excess length of the longest streak (counted in flips, not as a share of the sequence), addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/periodic_ideal_absolute_streak_penalty.py`

### streak_penalized_personal_prototype — rank 4, 8.7 ± 5.7 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -1078.6

**Hypothesis:** Refinement of `person_specific_alternation_prototype` (local representativeness with a person-specific preferred alternation rate): people judge randomness by multiscale local H/T balance and by irregularity — how far a sequence's switching rate is from their own preferred rate, plus a periodic-template penalty — and, as the one added component, they treat the longest streak in a sequence as a salient sign of non-randomness, penalising long runs beyond what the overall switching rate implies. (The distance from the personal preferred rate is taken as smooth and squared, as in the incumbent, rather than absolute; the claim is the added streak penalty.) This targets the critique that the best model under-penalises long streaks and regular patterns once alternation rate is accounted for.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/streak_penalized_personal_prototype.py`

### length_scaled_alternation_ideal_streak — rank 5, 13.4 ± 6.7 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1083.3

**Hypothesis:** Refinement of `personal_alternation_ideal_streak_penalty`: each person judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, with the longest streak counted as a further sign of non-randomness — but the alternation impression is treated as evidence that accumulates over the flips seen, so a given departure from the personal ideal weighs more heavily in a longer sequence (it rests on more transitions) than in a short one. The one change is this length-scaled sensitivity to the alternation-rate distance (sensitivity grows as a fitted power of the number of transitions), addressing the critique that people's preference for the higher-switch sequence falls off with length far less than a fixed-sensitivity, rate-based model predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/length_scaled_alternation_ideal_streak.py`

### personal_alternation_ideal_streak_penalty — rank 6, 14.3 ± 6.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1084.2

**Hypothesis:** Refinement of `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but people also treat the longest streak of identical outcomes as a salient sign of non-randomness in its own right. The one change is a shared penalty on the sequence's longest run (relative to its length), so that of two sequences equally close to a person's ideal switching rate, the one containing a longer streak looks less random — addressing the critique that long streaks are penalised beyond what the alternation rate explains.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_alternation_ideal_streak_penalty.py`

### personal_alternation_ideal — rank 7, 18.3 ± 8.4 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1088.2

**Hypothesis:** Each person carries their own ideal switching rate for a random coin — some expect a fair coin to flip side about half the time, others expect it to alternate far more often — and judges a sequence as random to the extent that its proportion of H/T switches is close to that personal ideal. The current best model assumes one shared alternation prototype for everyone; this model disagrees most sharply on pairs where both sequences alternate heavily (e.g. HTHTHTHT versus HTHHTHTH), which strong over-alternators should split decisively toward the more alternating sequence while moderate people choose the other.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_alternation_ideal.py`

### pair_normalized_alternation_contrast — rank 8, 36.1 ± 10.2 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1106.0

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the pair relative to itself. Each person compares how far each sequence's switching rate is from their own ideal switching rate, but the difference between the two is weighed relative to how far both sequences are from that ideal together (divisive contrast normalisation, as in Weber's law): a given gap decides the choice sharply when both sequences are near the ideal and barely matters when both are far from it, so the same sequence is chosen more or less decisively depending on its partner.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pair_normalized_alternation_contrast.py`

### person_specific_alternation_prototype — rank 9, 57.8 ± 14.8 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1127.7

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge randomness by local H/T balance across scales plus irregularity measured as distance from a preferred alternation rate and a periodic-template penalty, but each person holds their own preferred alternation rate (prototype) rather than one shared by everyone. The single change is making the alternation prototype person-specific, drawn from a population distribution, which addresses the critique that the model under-produces individual differences in preference for alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/person_specific_alternation_prototype.py`

### motif_stack_person_alternation — rank 10, 91.3 ± 30.0 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1161.2

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the likelihood ratio of a fair coin against Griffiths et al.'s four-motif stack automaton, a regularity detector shared by everyone (held at the automaton settings `motif_stack` itself estimated on these data), but individuals differ in how strongly they additionally treat frequent alternation itself as a sign of randomness. The one change is a person-specific alternation-preference weight, drawn from a population distribution, on the difference in alternation rate between the two sequences — addressing the critique that people differ far more in their preference for alternation than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/motif_stack_person_alternation.py`

### person_alternation_prototype — rank 11, 91.6 ± 18.5 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1161.5

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky local representativeness: a sequence looks random when it is locally balanced and irregular, irregularity being closeness to an over-alternating prototype plus a periodic-template penalty). The one change: each person carries their own ideal alternation rate (prototype), drawn from a population distribution, rather than everyone sharing a single prototype — so people differ in how much alternation they expect from a random coin, which the single-population incumbent cannot produce (the critique's large across-participant spread in preference for the more-alternating sequence).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/person_alternation_prototype.py`

### bayes_pattern_vs_alternating_coin — rank 12, 129.8 ± 24.0 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1199.7

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent it is better explained by a random coin than by a noisy repeating pattern (a short template of period 1-4, such as HHHH, HTHT or HHTT, copied with occasional errors). The single distortion is in their picture of the random coin: each person believes a fair coin switches sides at their own rate (often more than half the time), so alternation counts as evidence for randomness up to the point where the sequence becomes regular enough to be better explained by a repeating template.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/bayes_pattern_vs_alternating_coin.py`

### personal_markov_coin_belief — rank 13, 182.0 ± 33.7 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1251.9

**Hypothesis:** Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/personal_markov_coin_belief.py`

### most_salient_regular_stretch — rank 14, 277.4 ± 31.1 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1347.3

**Hypothesis:** People judge a sequence by its single most striking regular stretch: they scan it for the longest streak of identical flips and the longest stretch of perfect alternation, and whichever of these looks most patterned (a smooth maximum of the two, not a sum over the whole sequence) decides how non-random the sequence seems; they choose the sequence whose most striking stretch is less salient. Everyone finds both kinds of stretch a sign of pattern, and people share how salient a streak is but differ in how salient a stretch of strict alternation is (for some it is a glaring pattern, for others it barely registers, so for them a long streak decides almost every choice).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/most_salient_regular_stretch.py`

### position_weighted_switch_impression — rank 15, 288.5 ± 43.7 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1358.4

**Hypothesis:** People do not weigh every part of a sequence equally: they form their impression of how often the coin switches mainly from one end of the sequence (the last flips they read, or the first), and judge a sequence as more random the closer that position-weighted switch rate comes to a shared ideal switch rate for a random coin. Two sequences with the same overall number of switches can therefore be judged differently depending on whether their switches or their repeats sit at the attended end.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/position_weighted_switch_impression.py`

### local_representativeness — rank 16, 310.5 ± 42.0 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1380.4

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/local_representativeness.py`

### motif_stack — rank 17, 357.5 ± 46.9 nats behind the best (7.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1427.4

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/motif_stack.py`

### context_learner_surprise — rank 18, 489.1 ± 64.8 nats behind the best (7.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1559.0

**Hypothesis:** People read a sequence flip by flip and, with a short memory of the last two flips, keep trying to predict the next flip from what followed the same two-flip context earlier in the sequence (an online pattern learner); a sequence looks random to the extent that these running predictions keep failing, i.e. by the average surprise the learner experiences. Long streaks and repeating patterns (HTHTHTHT, HHTTHHTT) quickly become predictable to such a learner and so look non-random even when their overall switch rate is near what people expect, and people differ in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/context_learner_surprise.py`

### falk_konold_dp — rank 19, 494.6 ± 45.3 nats behind the best (10.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1564.5

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/falk_konold_dp.py`

### best_lag_copy_rule_detector — rank 20, 586.8 ± 64.1 nats behind the best (9.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1656.7

**Hypothesis:** People judge randomness by searching each sequence for a simple copy rule — "each flip repeats the flip k places back" for some small lag k (lag 1 catches streaks, lag 2 catches HTHT alternation, lag 4 catches HHTTHHTT) — and a sequence looks non-random to the extent that its best such rule predicts its flips. They choose the sequence whose best copy rule fits worse, with people differing only in how strongly that detected regularity drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/best_lag_copy_rule_detector.py`

### finite_experience_occurrence — rank 21, 633.3 ± 67.5 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1703.2

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/finite_experience_occurrence.py`

### running_tally_drift_ideal — no comparison row

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails (or tails lead heads), and judge randomness by how far that tally wanders from balance along the way: each person expects a fair coin's tally to drift away from even by some typical amount (scaled to how many flips have been read so far), and a sequence looks random to the extent that its average drift matches that personal expectation. A long streak makes the tally run away from balance (too much drift), while strict alternation pins it at even (too little drift), so the path of the tally — not the final count or the switch rate — decides the choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/running_tally_drift_ideal.py`

### exemplar_designed_pattern_similarity — no comparison row

**Hypothesis:** People judge randomness by exemplar memory: they hold remembered examples of "designed" coin sequences — streaks (HHHH), strict alternation (HTHT), and short repeating units (HHT, HHTT, HHHT, HTTT and their shifts and mirror images) — and a sequence looks non-random to the extent that it is similar to these stored patterns, with similarity falling off exponentially with the number of flips that would have to change to turn the sequence into a stored pattern (summed over all stored examples, as in a generalized context model). Random examples are remembered too diffusely to favour any particular sequence, so people pick the sequence with less summed similarity to the designed exemplars; individuals differ in how strongly the alternation exemplars are stored, so for some people near-alternating sequences look designed and for others they do not.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/exemplar_designed_pattern_similarity.py`

### gamblers_fallacy_fading_memory — no comparison row

**Hypothesis:** People judge randomness by reading a sequence flip by flip with a gambler's-fallacy expectation: after each flip they expect the coin to "balance out" against the flips they have just seen, with recent flips weighing more than earlier ones (a fading memory), and a sequence looks random to the extent that its flips keep matching these balancing expectations (its probability under that self-correcting subjective coin). Because the expectation of a switch builds up the longer a streak continues, a long run is increasingly surprising beyond what its number of switches implies; people differ in how strongly they hold this balancing expectation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/gamblers_fallacy_fading_memory.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
