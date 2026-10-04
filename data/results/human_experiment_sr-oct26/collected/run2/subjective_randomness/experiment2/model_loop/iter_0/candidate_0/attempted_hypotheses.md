# Tried before

36 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### personal_markov_coin_belief — pruned (experiment1 end of experiment)

**Outcome:** 205.7 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

### personal_alternation_ideal — pruned (experiment1 end of experiment)

**Outcome:** 42.1 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Each person carries their own ideal switching rate for a random coin — some expect a fair coin to flip side about half the time, others expect it to alternate far more often — and judges a sequence as random to the extent that its proportion of H/T switches is close to that personal ideal. The current best model assumes one shared alternation prototype for everyone; this model disagrees most sharply on pairs where both sequences alternate heavily (e.g. HTHTHTHT versus HTHHTHTH), which strong over-alternators should split decisively toward the more alternating sequence while moderate people choose the other.

### personal_switch_rate_ideal — rejected (experiment1 round 0 candidate 2 lens 2)

**Outcome:** predicts like existing model 'personal_alternation_ideal' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of personal_alternation_ideal, not a new hypothesis.

**Hypothesis:** Each person carries their own internal ideal of how often a random coin should switch between heads and tails, and judges a sequence as more random the closer its switch rate comes to that personal ideal. People differ in where this ideal sits (some expect heavy alternation, others near-even switching), so the same pair can be judged in opposite directions by different people.

### person_alternation_prototype — pruned (experiment1 end of experiment)

**Outcome:** 115.3 nats behind graded_periodicity_personal_ideal (6.4× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky local representativeness: a sequence looks random when it is locally balanced and irregular, irregularity being closeness to an over-alternating prototype plus a periodic-template penalty). The one change: each person carries their own ideal alternation rate (prototype), drawn from a population distribution, rather than everyone sharing a single prototype — so people differ in how much alternation they expect from a random coin, which the single-population incumbent cannot produce (the critique's large across-participant spread in preference for the more-alternating sequence).

### person_specific_alternation_prototype — pruned (experiment1 end of experiment)

**Outcome:** 81.5 nats behind graded_periodicity_personal_ideal (4.9× dse)

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge randomness by local H/T balance across scales plus irregularity measured as distance from a preferred alternation rate and a periodic-template penalty, but each person holds their own preferred alternation rate (prototype) rather than one shared by everyone. The single change is making the alternation prototype person-specific, drawn from a population distribution, which addresses the critique that the model under-produces individual differences in preference for alternation.

### motif_stack_person_alternation — pruned (experiment1 end of experiment)

**Outcome:** 115.0 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the likelihood ratio of a fair coin against Griffiths et al.'s four-motif stack automaton, a regularity detector shared by everyone (held at the automaton settings `motif_stack` itself estimated on these data), but individuals differ in how strongly they additionally treat frequent alternation itself as a sign of randomness. The one change is a person-specific alternation-preference weight, drawn from a population distribution, on the difference in alternation rate between the two sequences — addressing the critique that people differ far more in their preference for alternation than a single-population model produces.

### position_weighted_switch_impression — pruned (experiment1 end of experiment)

**Outcome:** 312.2 nats behind graded_periodicity_personal_ideal (7.2× dse)

**Hypothesis:** People do not weigh every part of a sequence equally: they form their impression of how often the coin switches mainly from one end of the sequence (the last flips they read, or the first), and judge a sequence as more random the closer that position-weighted switch rate comes to a shared ideal switch rate for a random coin. Two sequences with the same overall number of switches can therefore be judged differently depending on whether their switches or their repeats sit at the attended end.

### context_learner_surprise — pruned (experiment1 end of experiment)

**Outcome:** 512.8 nats behind graded_periodicity_personal_ideal (7.7× dse)

**Hypothesis:** People read a sequence flip by flip and, with a short memory of the last two flips, keep trying to predict the next flip from what followed the same two-flip context earlier in the sequence (an online pattern learner); a sequence looks random to the extent that these running predictions keep failing, i.e. by the average surprise the learner experiences. Long streaks and repeating patterns (HTHTHTHT, HHTTHHTT) quickly become predictable to such a learner and so look non-random even when their overall switch rate is near what people expect, and people differ in how strongly this felt surprise drives their choice.

### bayes_pattern_vs_alternating_coin — pruned (experiment1 end of experiment)

**Outcome:** 153.5 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent it is better explained by a random coin than by a noisy repeating pattern (a short template of period 1-4, such as HHHH, HTHT or HHTT, copied with occasional errors). The single distortion is in their picture of the random coin: each person believes a fair coin switches sides at their own rate (often more than half the time), so alternation counts as evidence for randomness up to the point where the sequence becomes regular enough to be better explained by a repeating template.

### best_lag_copy_rule_detector — pruned (experiment1 end of experiment)

**Outcome:** 610.5 nats behind graded_periodicity_personal_ideal (9.4× dse)

**Hypothesis:** People judge randomness by searching each sequence for a simple copy rule — "each flip repeats the flip k places back" for some small lag k (lag 1 catches streaks, lag 2 catches HTHT alternation, lag 4 catches HHTTHHTT) — and a sequence looks non-random to the extent that its best such rule predicts its flips. They choose the sequence whose best copy rule fits worse, with people differing only in how strongly that detected regularity drives their choice.

### periodic_penalized_alternation_ideal — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** Refinement of the incumbent `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but in addition everyone notices when a sequence is built by repeating a short unit (e.g. HTHTHTHT, HHTTHHTT, HTTHTTHT) and counts that visible periodic pattern against its randomness. The one change is this shared periodic-pattern penalty, addressing the critique that people reject regular repeating patterns (and perfect alternation) far more than the alternation-distance mechanism alone predicts.

### personal_alternation_ideal_streak_penalty — pruned (experiment1 end of experiment)

**Outcome:** 38.0 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** Refinement of `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but people also treat the longest streak of identical outcomes as a salient sign of non-randomness in its own right. The one change is a shared penalty on the sequence's longest run (relative to its length), so that of two sequences equally close to a person's ideal switching rate, the one containing a longer streak looks less random — addressing the critique that long streaks are penalised beyond what the alternation rate explains.

### streak_penalized_personal_prototype — pruned (experiment1 end of experiment)

**Outcome:** 32.4 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Refinement of `person_specific_alternation_prototype` (local representativeness with a person-specific preferred alternation rate): people judge randomness by multiscale local H/T balance and by irregularity — how far a sequence's switching rate is from their own preferred rate, plus a periodic-template penalty — and, as the one added component, they treat the longest streak in a sequence as a salient sign of non-randomness, penalising long runs beyond what the overall switching rate implies. (The distance from the personal preferred rate is taken as smooth and squared, as in the incumbent, rather than absolute; the claim is the added streak penalty.) This targets the critique that the best model under-penalises long streaks and regular patterns once alternation rate is accounted for.

### pair_normalized_alternation_contrast — pruned (experiment1 end of experiment)

**Outcome:** 59.8 nats behind graded_periodicity_personal_ideal (4.6× dse)

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the pair relative to itself. Each person compares how far each sequence's switching rate is from their own ideal switching rate, but the difference between the two is weighed relative to how far both sequences are from that ideal together (divisive contrast normalisation, as in Weber's law): a given gap decides the choice sharply when both sequences are near the ideal and barely matters when both are far from it, so the same sequence is chosen more or less decisively depending on its partner.

### most_salient_regular_stretch — pruned (experiment1 end of experiment)

**Outcome:** 301.1 nats behind graded_periodicity_personal_ideal (9.6× dse)

**Hypothesis:** People judge a sequence by its single most striking regular stretch: they scan it for the longest streak of identical flips and the longest stretch of perfect alternation, and whichever of these looks most patterned (a smooth maximum of the two, not a sum over the whole sequence) decides how non-random the sequence seems; they choose the sequence whose most striking stretch is less salient. Everyone finds both kinds of stretch a sign of pattern, and people share how salient a streak is but differ in how salient a stretch of strict alternation is (for some it is a glaring pattern, for others it barely registers, so for them a long streak decides almost every choice).

### personal_ideal_with_personal_lapse — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** People judge which sequence looks more random by how close its proportion of H/T switches is to their own personal ideal switching rate, but the decision rule is not always engaged: each person has their own lapse rate, the share of trials on which they do not evaluate the sequences at all and pick a side at random. People therefore differ not only in what they think randomness looks like but in how consistently they act on it, so even a pair with a clear winner is split by some participants, more for some people than for others.

### length_scaled_alternation_ideal — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate, with a shared penalty for visibly periodic sequences, but the weight people give to a deviation from their ideal grows (or shrinks) with how many flips they have seen — a switch rate read off a long sequence is treated as stronger evidence than the same rate in a short one. The one change is a fitted power-law scaling of the alternation-distance sensitivity with the number of transitions in the sequence, addressing the critique that the incumbent mispredicts how the preference for the more-switching sequence changes with sequence length.

### length_scaled_alternation_ideal_2 — rejected (experiment1 round 2 candidate 4 refine incumbent periodic_penalized_alternation_ideal)

**Outcome:** predicts like existing model 'length_scaled_alternation_ideal' (p_left RMSE 0.00011 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_scaled_alternation_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, with a shared penalty for visibly periodic sequences, but the evidence from switch rate accumulates with the number of transitions people see — a switch-rate deviation observed over seven transitions is a more convincing sign of non-randomness than the same deviation over two. The one change is that the sensitivity to distance from the personal ideal scales as a fitted power of the number of transitions in the sequence, addressing the critique that the model under-predicts how the preference for the more-switching sequence holds up as sequences get longer.

### length_scaled_alternation_ideal_streak — pruned (experiment1 end of experiment)

**Outcome:** 37.1 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of `personal_alternation_ideal_streak_penalty`: each person judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, with the longest streak counted as a further sign of non-randomness — but the alternation impression is treated as evidence that accumulates over the flips seen, so a given departure from the personal ideal weighs more heavily in a longer sequence (it rests on more transitions) than in a short one. The one change is this length-scaled sensitivity to the alternation-rate distance (sensitivity grows as a fitted power of the number of transitions), addressing the critique that people's preference for the higher-switch sequence falls off with length far less than a fixed-sensitivity, rate-based model predicts.

### periodic_ideal_absolute_streak_penalty — pruned (experiment1 end of experiment)

**Outcome:** 26.2 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, with a shared penalty for visibly periodic sequences, but people also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness, however long the sequence. The one change is this shared penalty on the absolute excess length of the longest streak (counted in flips, not as a share of the sequence), addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

### running_tally_drift_ideal — pruned (experiment1 end of experiment)

**Outcome:** 234.8 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails (or tails lead heads), and judge randomness by how far that tally wanders from balance along the way: each person expects a fair coin's tally to drift away from even by some typical amount (scaled to how many flips have been read so far), and a sequence looks random to the extent that its average drift matches that personal expectation. A long streak makes the tally run away from balance (too much drift), while strict alternation pins it at even (too little drift), so the path of the tally — not the final count or the switch rate — decides the choice.

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Outcome:** 668.3 nats behind graded_periodicity_personal_ideal (10.6× dse)

**Hypothesis:** People judge randomness by exemplar memory: they hold remembered examples of "designed" coin sequences — streaks (HHHH), strict alternation (HTHT), and short repeating units (HHT, HHTT, HHHT, HTTT and their shifts and mirror images) — and a sequence looks non-random to the extent that it is similar to these stored patterns, with similarity falling off exponentially with the number of flips that would have to change to turn the sequence into a stored pattern (summed over all stored examples, as in a generalized context model). Random examples are remembered too diffusely to favour any particular sequence, so people pick the sequence with less summed similarity to the designed exemplars; individuals differ in how strongly the alternation exemplars are stored, so for some people near-alternating sequences look designed and for others they do not.

### gamblers_fallacy_fading_memory — pruned (experiment1 end of experiment)

**Outcome:** 136.3 nats behind graded_periodicity_personal_ideal (4.1× dse)

**Hypothesis:** People judge randomness by reading a sequence flip by flip with a gambler's-fallacy expectation: after each flip they expect the coin to "balance out" against the flips they have just seen, with recent flips weighing more than earlier ones (a fading memory), and a sequence looks random to the extent that its flips keep matching these balancing expectations (its probability under that self-correcting subjective coin). Because the expectation of a switch builds up the longer a streak continues, a long run is increasingly surprising beyond what its number of switches implies; people differ in how strongly they hold this balancing expectation.

### balance_aware_length_scaled_ideal — rejected (experiment1 round 3 candidate 3 refine incumbent length_scaled_alternation_ideal)

**Outcome:** predicts like existing model 'length_scaled_alternation_ideal' (p_left RMSE 0.00111 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of length_scaled_alternation_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with sensitivity scaling as a power of the number of transitions) and with a shared penalty for visibly periodic sequences, but people also expect a random coin to give roughly equal numbers of heads and tails, so a lopsided H/T count (e.g. HHHHTT versus HHHTTT, which have the same number of switches) counts against randomness. The one change is a shared penalty on the H/T imbalance of each sequence (the squared deviation of its share of heads from one half, grafted from the global-balance component of `local_representativeness`), addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run — which, at equal switch counts, is the more balanced one.

### person_sensitivity_length_scaled_ideal — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a length-scaled sensitivity and a shared penalty for visibly periodic sequences), but people also differ in how decisively they act on that impression — some pick the sequence nearer their ideal almost every time, others only weakly lean toward it. The one change is a person-specific sensitivity to the alternation-distance, drawn from a population distribution, so that how sharply a person discriminates is an individual trait just as their ideal switching rate is.

### personal_lapse_ideal_streak_excess — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** Refinement of `personal_ideal_with_personal_lapse`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, and still has their own lapse rate (trials on which they pick a side at random), but when they do evaluate the sequences they also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness. The one change is this shared absolute-streak penalty inside the engaged decision, addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

### local_window_alternation_ideal — pruned (experiment1 end of experiment)

**Outcome:** 61.1 nats behind graded_periodicity_personal_ideal (4.8× dse)

**Hypothesis:** Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its switching to their own ideal switching rate is (sensitivity scaling as a power of the number of transitions, with a shared penalty for visibly periodic sequences), but they apply that ideal locally rather than to the sequence as a whole — they check every short stretch of three flips against their ideal and average how far each stretch is from it. The one change is this local reading of the alternation ideal: a sequence whose switches are bunched together, leaving a long streak with no switching, contains stretches far from the ideal and so looks less random than a sequence with the same total number of switches spread evenly, addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

### chunk_entropy_variety — pruned (experiment1 end of experiment)

**Outcome:** 246.4 nats behind graded_periodicity_personal_ideal (4.1× dse)

**Hypothesis:** People judge randomness by the variety of short patterns a sequence contains: reading it as overlapping chunks of one to four flips, a sequence looks random to the extent that its chunks of each size are spread evenly over the different possible chunks (high chunk entropy for its length), and repetitive to the extent that the same few chunks recur — through a lopsided H/T count, a long streak, strict alternation, or an almost-repeating unit with one slip. People differ in how strongly this felt variety drives their choice. This disagrees most sharply with the switch-rate-ideal incumbent on pairs with similar switch counts where one sequence reuses its chunks (near-periodic sequences, or one with a long streak), and on block-structured sequences like HHTTHHTT, whose middling switch rate the incumbent rates as plausible but whose chunk variety is low.

### leaky_switch_impression_tracking — pruned (experiment1 end of experiment)

**Outcome:** 108.1 nats behind graded_periodicity_personal_ideal (5.1× dse)

**Hypothesis:** People read a sequence flip by flip and keep a running impression of how often the coin is switching, updated after every flip by a leaky (exponentially forgetting) memory in which recent transitions weigh more than earlier ones; at every point of the reading they compare this running impression with their own ideal switching rate, and the sequence looks random to the extent that the impression stays close to that ideal throughout. Because the impression tracks the reading moment by moment, a long streak drags it far below the ideal for several flips (and a stretch of strict alternation pushes it above), so a sequence with bunched switches looks less random than one with the same number of switches spread evenly; people differ in their ideal switching rate, while the memory's forgetting rate is shared.

### graded_periodicity_personal_ideal — dropped (experiment2)

**Outcome:** non-finite ELPD-LOO (nan) on this experiment's data

**Hypothesis:** Refinement of the incumbent `person_sensitivity_length_scaled_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a person-specific, length-scaled sensitivity), and everyone still counts a visibly repeating pattern against randomness — but the pattern detector is graded rather than all-or-nothing: people also notice a sequence that *almost* repeats a short unit (period 1–4) with one or two flips out of place, and the penalty fades smoothly with the number of flips that break the pattern. The one change is replacing the strict periodic indicator with this graded near-periodicity penalty (shared strength, shared fall-off per mismatch), addressing the critique that people avoid almost-repeating patterns that the incumbent's strict check misses.

### heads_lean_person_sensitivity_ideal — rejected (experiment1 round 4 candidate 4 refine incumbent person_sensitivity_length_scaled_ideal)

**Outcome:** predicts like existing model 'heads_default_alternation_ideal' (p_left RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heads_default_alternation_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `person_sensitivity_length_scaled_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a person-specific, length-scaled sensitivity and a shared penalty for visibly periodic sequences), but people are not symmetric between heads and tails — heads is the outcome they think of first, so a sequence in which heads makes up more of the flips reads as more like a typical coin sequence and looks more random than its mirror image. The one change is a shared lean toward the sequence with the larger share of heads, addressing the critique that among pairs with equal switch counts people choose the heads-heavier sequence more often than the H/T-symmetric incumbent predicts.

### streak_excess_person_sensitivity_ideal — rejected (experiment1 round 4 candidate 4 refine incumbent person_sensitivity_length_scaled_ideal repair 1)

**Outcome:** predicts like existing model 'person_sensitivity_length_scaled_ideal' (p_left RMSE 0.00015 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of person_sensitivity_length_scaled_ideal, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `person_sensitivity_length_scaled_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate (with a person-specific sensitivity that scales with the number of transitions, and a shared penalty for visibly periodic sequences), but people also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness, however the switches are spread elsewhere. The one change is this shared absolute-streak-excess penalty added to the incumbent's judgement, addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

### falk_konold_dp — pruned (experiment1 end of experiment)

**Outcome:** 518.3 nats behind graded_periodicity_personal_ideal (11.3× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

### motif_stack — pruned (experiment1 end of experiment)

**Outcome:** 381.2 nats behind graded_periodicity_personal_ideal (8.2× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Outcome:** 657.0 nats behind graded_periodicity_personal_ideal (9.7× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

### local_representativeness — pruned (experiment1 end of experiment)

**Outcome:** 334.3 nats behind graded_periodicity_personal_ideal (8.0× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.
