# Refinement menu

The models you may refine, other than the incumbent `personal_complement_symmetry_aversion`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### personal_memory_capacity_blur — rank 1, 0.8 ± 6.1 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -3466.5

**Hypothesis:** To compare two sequences people must hold both in working memory, and each person has their own memory capacity: once a sequence's length approaches or exceeds that capacity, the flips are encoded noisily and the comparison is blurred, so a low-capacity person follows their impression of which sequence looks more random sharply for short pairs but only weakly for long ones, while a high-capacity person is equally decisive at every length. The impression itself is the current best judgement (personal ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety, the most lopsided stretch and complement symmetry, and a personal guessing rate); the personal capacity predicts that people differ in how consistently they agree with the majority, and that these differences are concentrated on long pairs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_memory_capacity_blur.py`

### length_switching_complement_symmetry — rank 2, 4.6 ± 4.6 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3470.2

**Hypothesis:** Refinement of `length_dependent_ideal_switching` (rank 1): people still apply the law of small numbers to switching — the switching rate they consider ideal for a random coin shifts with how many flips they are looking at, by a shared amount on top of each person's own ideal — and still judge each sequence by their personal sensitivity to that ideal (lenient toward over-alternation), personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, with a personal guessing rate. The one change is grafting in the incumbent's complement-symmetry aversion: a sequence of four or more flips that equals its own reverse with heads and tails swapped (HHTT, HHHTHTTT, HTHTHTHT) looks deliberately constructed and so less random, by a shared weight — so the length-dependent ideal is estimated on top of the best current judgement, predicting that on short versus long pairs differing mainly in switching rate the preferred amount of alternation shifts with length even after antisymmetric sequences are penalised.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/length_switching_complement_symmetry.py`

### complement_symmetry_aversion — rank 3, 4.8 ± 4.3 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3470.5

**Hypothesis:** People notice complement symmetry (antisymmetry): a sequence of four or more flips whose second half is the first half read backwards with heads and tails swapped — it equals its own reverse with H and T exchanged, as in HHTT, TTTTHHHH, HHTTHHTT, HHHTHTTT or strict alternation HTHTHTHT — looks deliberately constructed, and so less random, by a shared weight. This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate), which registers mirror symmetry (palindromes) but not this second kind of symmetry, so it predicts people reject antisymmetric sequences more often than the best model does even when their switching rate, balance and streaks look acceptable.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/complement_symmetry_aversion.py`

### length_dependent_ideal_switching — rank 4, 27.4 ± 9.0 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3493.1

**Hypothesis:** Law of small numbers applied to switching: people expect even a short run of coin flips to be representative of a random coin, and the switching rate they consider ideal depends on how many flips they are looking at — a shared shift with sequence length moves everyone's personal ideal switching rate, so short sequences are held to a different (e.g. more alternating) ideal than long ones. Each sequence's randomness is otherwise judged as in the current best model (personal ideal and sensitivity with lenient treatment of over-alternation, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate). The disagreement with the best model lives in the length dimension: on short pairs (2–4 flips) versus long pairs (7–8 flips) that differ mainly in switching rate, this model predicts the preferred amount of alternation flips or strengthens with length, whereas the best model applies one length-blind ideal.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/length_dependent_ideal_switching.py`

### personal_decisiveness_complement_symmetry — rank 5, 32.4 ± 14.5 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3498.1

**Hypothesis:** Refinement of the incumbent `complement_symmetry_aversion`: each sequence's apparent randomness is still judged by its switching rate against a personal ideal (lenient toward over-alternation), personal weights on heads/tails imbalance and the longest streak, and shared weights on the final streak, mirror symmetry, complement symmetry, run-length variety and the most lopsided stretch. The one change is where people's consistency comes from: instead of a personal sensitivity to the switching rate alone and a personal guessing rate, each person has their own overall decisiveness (drawn from a population) that sharpens or flattens the whole comparison of the two impressions, with one shared guessing rate — so some people follow every cue almost deterministically while others follow all of them weakly, addressing the critique (surviving FDR) that people differ in how often they agree with the majority choice more than the incumbent produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_decisiveness_complement_symmetry.py`

### side_bias_asymmetric_tolerance_stretch — rank 6, 33.8 ± 10.1 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3499.4

**Hypothesis:** Refinement of `personal_side_bias_lopsided_stretch` (rank 2): each person still has a habitual leaning toward the left or right response button, drawn from a population that may lean left overall, which tips every comparison they make toward that side by a fixed amount, and each sequence is still judged by the most lopsided stretch on top of a personal ideal switching rate and sensitivity, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is grafting in the incumbent's asymmetric tolerance: switching more often than one's ideal costs only a shared fraction of what the same shortfall in switching costs — so the button leaning (the critique's excess spread of people's left-choice rates) is estimated on top of the best current judgement of randomness rather than on a weaker one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/side_bias_asymmetric_tolerance_stretch.py`

### asymmetric_tolerance_lopsided_stretch — rank 7, 34.6 ± 9.8 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3500.3

**Hypothesis:** Refinement of the incumbent `most_lopsided_stretch_aversion`, grafting in the asymmetric ideal-rate tolerance of `run_variety_asymmetric_ideal_rate`: people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, and still judge it by their own ideal switching rate and sensitivity, personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is that a departure from one's ideal switching rate counts differently in the two directions: switching more often than the ideal costs only a shared fraction (fitted, possibly far from one) of what the same shortfall in switching (streakiness) costs, so people are lenient toward over-alternation — predicting that, beyond what the local-excess cue explains, the more alternating of two highly alternating sequences is chosen more often than a symmetric penalty implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/asymmetric_tolerance_lopsided_stretch.py`

### length_weighted_ideal_rate_lopsided — rank 8, 35.8 ± 9.8 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3501.5

**Hypothesis:** Refinement of the incumbent `asymmetric_tolerance_lopsided_stretch`: people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, still compare its switching rate with their own ideal (with their own sensitivity, judging switching too often more leniently than too rarely by a shared fraction), and still use personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is that a departure from one's ideal switching rate is weighed as evidence that grows (or shrinks) with the number of flip-to-flip transitions it is observed over — a shared power of sequence length — so the same off-ideal switching rate is shrugged off in a short sequence and taken seriously in a long one (or the reverse, if the fitted exponent is negative), while the streak, balance, symmetry and local-excess cues keep their length-blind weights; this predicts that the alternation preference is weaker for short pairs and stronger for long pairs than the incumbent's length-blind ideal-rate penalty implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/length_weighted_ideal_rate_lopsided.py`

### satisficing_first_look_acceptance — rank 9, 42.2 ± 10.2 nats behind the best (4.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3507.8

**Hypothesis:** People do not always compare the two sequences: they satisfice. A person looks first at one sequence (each person has their own habitual tendency to start with the left or the right one) and, if it already looks random enough by their own personal standard, simply picks it without weighing the other; only when the first sequence fails that standard do they compare the two and choose the one that looks more random (each sequence's randomness judged as in the current best model, and with occasional guessing). So side preferences are not a fixed tilt: they come from lenient people who stop at the first acceptable sequence, they are strongest when the first-inspected sequence looks reasonably random on its own, and strict people show almost none.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/satisficing_first_look_acceptance.py`

### absolute_local_excess_aversion — rank 10, 61.2 ± 16.3 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3526.8

**Hypothesis:** Refinement of the incumbent `asymmetric_tolerance_lopsided_stretch`: people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, and still judge it by their own ideal switching rate and sensitivity (a departure toward switching too often costing a shared, fitted multiple of what the same shortfall costs), personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is how the local excess is sized: people notice it in absolute flips — a surplus of three heads in one stretch is equally striking whether the sequence has four flips or eight — rather than as a share of the sequence's length, so the local-excess penalty bites harder in long sequences and much less in short ones than the incumbent's length-normalised cue implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/absolute_local_excess_aversion.py`

### default_side_guessing — rank 11, 64.2 ± 15.3 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3529.9

**Hypothesis:** The change is in how people guess. On the trials where a person does not actually compare the two sequences, they do not flip a mental coin. They fall back on a habitual default button, and how strongly each person leans toward Left (the sequence read first) or Right is a stable trait drawn from a population that may lean Left overall. On trials where they do compare, they judge randomness exactly as the current best model does, with no side bias. So side preferences show up only through guessing: they are strongest in people who guess often, and the more decisive pairs do not cancel them out.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/default_side_guessing.py`

### personal_side_bias_lopsided_stretch — rank 12, 65.0 ± 15.6 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3530.6

**Hypothesis:** Refinement of the incumbent `most_lopsided_stretch_aversion`: each person still judges how random each sequence looks exactly as the incumbent does (personal ideal switching rate and sensitivity, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the single most lopsided stretch, and a personal guessing rate). The one change is in the decision: each person also has a habitual leaning toward the left or right response button, drawn from a population that may lean left overall, which tips every comparison they make toward that side by a fixed amount — so it decides near-ties, and people's own left-choice rates differ more than the stimuli alone produce (the critique's participant side-bias spread and overall left-choice rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_side_bias_lopsided_stretch.py`

### most_lopsided_stretch_aversion — rank 13, 65.5 ± 15.3 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3531.2

**Hypothesis:** People notice the single most lopsided stretch of a sequence — the contiguous run of flips in which heads most outnumber tails (or tails most outnumber heads), even when it is not one unbroken streak (HHTHH is a three-heads excess) — and that one striking local excess makes the sequence look less random by a shared weight. This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity to it, personal weights on overall heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate); because a strictly alternating sequence never builds a local excess beyond one flip while a slightly less alternating one with a repeat or two clustered together does, it predicts that among two highly alternating sequences people choose the more alternating one more often than the best model does (the critique's high-alternation discrepancy).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/most_lopsided_stretch_aversion.py`

### lead_persistence_aversion — rank 14, 67.0 ± 15.2 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3532.7

**Hypothesis:** People keep a running tally of heads versus tails as they read a sequence and track which side is "in the lead"; a fair coin, they expect, lets the lead change hands, so a sequence in which one side stays ahead for a long unbroken stretch of the path (HHTHTHTH: heads leads at every flip although no streak is longer than two and the counts are nearly even) looks biased and so less random, by a shared weight on the longest stretch of flips during which the tally never returns to balance. This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate), so it is the path's lead persistence, not how far the tally strays, that is added.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/lead_persistence_aversion.py`

### run_variety_asymmetric_ideal_rate — rank 15, 105.9 ± 22.4 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3571.5

**Hypothesis:** Refinement of the incumbent `run_length_variety_preference`: people still judge a sequence as more random when its runs of identical flips come in varied lengths rather than one repeated length, still compare its alternation rate with their own ideal switching rate (with their own sensitivity), still penalise heads/tails imbalance and the longest streak by personal weights, still weigh the final streak and mirror symmetry by shared weights, and still guess at a personal lapse rate. The one change is that departures from one's ideal switching rate are judged asymmetrically: switching more often than the ideal counts against a sequence by a shared fraction (fitted, possibly far below one) of what the same shortfall in switching (streakiness) costs — addressing the critique that among two highly alternating sequences people choose the more alternating one more often than the incumbent's symmetric quadratic ideal-rate penalty predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/run_variety_asymmetric_ideal_rate.py`

### longest_alternation_asymmetric_ideal — rank 16, 107.1 ± 22.6 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3572.8

**Hypothesis:** Refinement of `run_variety_asymmetric_ideal_rate` (rank 1, statistically tied with the best): people still compare a sequence's switching rate with their own ideal (with their own sensitivity, judging switching too often more leniently than too rarely by a shared fraction), still prefer varied run lengths, penalise heads/tails imbalance and the longest streak by personal weights, weigh the final streak and mirror symmetry by shared weights, and guess at a personal lapse rate. The one change is that people also notice the longest unbroken stretch of strict heads/tails alternation (HTHTH…) as a pattern in its own right — the alternation counterpart of the longest streak — and it shifts how random the sequence looks by a shared weight, so two sequences with the same number of switches differ when one gathers its switches into a long rigid alternating stretch (HHTHTHTT) and the other spreads them out (HTHHTTHT).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/longest_alternation_asymmetric_ideal.py`

### run_length_variety_preference — rank 17, 152.6 ± 28.1 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3618.3

**Hypothesis:** People judge randomness partly by the variety of a sequence's runs: a sequence whose runs of identical flips all have the same length (HTHTHTHT, HHTTHHTT, HHHTTTHH) has a regular, rhythmic run structure that looks designed, while one mixing runs of different lengths (HTTHHHTH) looks irregular and so random, by a shared weight on the number of distinct run lengths (relative to the most its length allows). This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity to it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and on mirror symmetry, and a personal guessing rate) and addresses the critique (surviving FDR) that among pairs matched on alternation rate people choose the sequence with more distinct run lengths more often than the best model predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/run_length_variety_preference.py`

### alignment_sharpened_comparison — rank 18, 153.1 ± 28.0 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3618.8

**Hypothesis:** People do not judge the two sequences one at a time: they lay them side by side and look for where they differ, so how decisively the difference in randomness drives the choice depends on how alignable the pair is. When the two sequences match flip for flip except in a few places (allowing heads and tails to be swapped), the differences that make one look less random stand out and the choice follows them sharply; when the two share little, there is no common frame and the same difference in apparent randomness decides the choice only weakly — so the same sequence is chosen at different rates beside a similar partner than beside a dissimilar one. Each sequence's randomness is still judged as in the current best model (personal ideal switching rate and sensitivity, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/alignment_sharpened_comparison.py`

### length_scaled_run_variety_load — rank 19, 154.2 ± 28.2 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3619.9

**Hypothesis:** Refinement of `length_load_encoding_noise` (rank 2): to compare two sequences people must hold both in working memory, and the number of flips changes how sharply the difference in apparent randomness decides the choice (a power of sequence length, by a shared fitted exponent: noise that grows with length, or evidence that accumulates with it); the impression itself is still each person's own ideal switching rate and sensitivity to it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and mirror symmetry, and people still guess on some trials at a personal lapse rate. The one change is grafting in the incumbent's run-length variety: a sequence whose runs of identical flips all share one length (HTHTHTHT, HHTTHHTT) looks rhythmic and designed, while one mixing runs of different lengths looks random, by a shared weight on the number of distinct run lengths relative to the most its length allows — so the length-dependent decisiveness is estimated on top of the best current impression, predicting that the incumbent's variety and alternation effects are weaker for short pairs and stronger for long ones than a length-blind comparison implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/length_scaled_run_variety_load.py`

### block_rhythm_regularity — rank 20, 188.6 ± 36.5 nats behind the best (5.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3654.3

**Hypothesis:** Refinement of the incumbent `run_length_variety_preference`: people still see a sequence made of runs that all share one length as rhythmic and designed, but the rhythm they notice is one of repeated blocks of identical flips (HHTTHHTT, HHHTTTHH) — the penalty is the lack of run-length variety scaled by the share of flips that sit in runs of two or more — so strict alternation (HTHTHTHT, every run a single flip) is no longer counted as a designed rhythm and is judged only against each person's own ideal switching rate. Everything else is the incumbent's judgement (personal ideal switching rate and sensitivity, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and mirror symmetry, a personal guessing rate); the one change addresses the critique that among two highly alternating sequences people choose the more alternating one more often than the incumbent predicts, which its variety term (zero variety for strict alternation) pushes against.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/block_rhythm_regularity.py`

### asymmetric_ideal_alternation_mirror — rank 21, 190.0 ± 33.9 nats behind the best (5.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3655.7

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate (with their own sensitivity), still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that the tolerance around the ideal is asymmetric: switching more often than one's ideal (over-alternation) counts against a sequence by a different, shared fraction of the penalty for switching less often (streakiness), so people can be far more lenient toward too much alternation than toward too little — addressing the critique (surviving FDR) that among two highly alternating sequences people choose the more alternating one more often than the incumbent's symmetric quadratic ideal-rate penalty predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/asymmetric_ideal_alternation_mirror.py`

### signed_personal_decisiveness — rank 22, 260.9 ± 34.0 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3726.5

**Hypothesis:** People differ in how they act on their impression of randomness along a single signed dimension of decisiveness: most people choose the sequence that looks more random to them with a personal strength, some barely follow their impression at all, and a few systematically pick the sequence that looks less random (reading the task in reverse). This one personal signed decisiveness replaces the current best model's personal guessing rates and personal weights on imbalance and the longest streak — the cues are weighed alike by everyone, and only each person's own ideal switching rate and how strongly (and in which direction) they act on the impression differ — so it predicts that people's agreement with the majority choice spreads far wider, down to below one half for some people, than guessing alone allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/signed_personal_decisiveness.py`

### personal_decisiveness_mirror_streak_lapse — rank 23, 298.8 ± 44.4 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3764.5

**Hypothesis:** Refinement of the incumbent `mirror_symmetry_streak_aversion_lapse`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that people differ in how decisively they act on their impression: each person has their own overall decisiveness (drawn from a population centred on the typical person) that sharpens or flattens every comparison they make, so some people consistently pick the more random-looking sequence while others, though not guessing, follow their impression only weakly — addressing the critique that participants differ in how often they agree with the majority choice more than the incumbent's person-level weights and lapse rates produce.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_decisiveness_mirror_streak_lapse.py`

### coin_belief_exemplar_cloud — rank 24, 1071.4 ± 73.2 nats behind the best (14.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4537.1

**Hypothesis:** People judge randomness by exemplar categorisation in the space of flip-by-flip similarity: their "random" category is a remembered cloud of same-length coin sequences, each as familiar as it is probable under a shared belief about how often a fair coin switches sides, and their "designed" category is a handful of rigid patterns (a solid streak, strict alternation, repeated pairs, repeated triples, two halves). A sequence looks random to the extent that its summed similarity to the random cloud outweighs its summed similarity to the designed patterns, with similarity falling off exponentially with the share of flips that differ, so a sequence is judged not by its own statistics but by how close it lies, flip for flip, to sequences an imagined coin would produce versus rule-made ones; people differ only in how decisively they act on this evidence.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/coin_belief_exemplar_cloud.py`

### bayesian_generator_personal_suspicion — rank 25, 1694.0 ± 90.7 nats behind the best (18.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -5159.7

**Hypothesis:** People judge randomness normatively, by Bayesian inference over generators: for each sequence they work out how probable it is that it came from a fair coin rather than from a non-random generator (a biased coin, a sticky-or-switchy coin, or a mirror- or complement-symmetric construction), and pick the sequence with the higher posterior probability of being random. The one distortion is a personal prior suspicion: each person brings their own prior belief that a sequence was made by a non-random generator, so a suspicious person's posteriors for most sequences are near zero and they discriminate only among the most random-looking ones, while a trusting person's are near one and they discriminate only among the most regular-looking ones — which makes people differ in how consistently they agree with the majority depending on which pairs they see.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/bayesian_generator_personal_suspicion.py`

### personal_first_read_anchoring — no comparison row

**Hypothesis:** People do not judge the two sequences independently: they read the left sequence first and its impression becomes an anchor, so each person weighs the first-read sequence's impression of randomness by a personal anchoring weight (more, or less, than the second sequence's), drawn from a population. The choice therefore depends not only on which sequence looks more random but on how random the pair looks overall relative to an ideal-looking sequence — when both look clearly non-random, a person who over-weights the first sequence's flaws turns to the right one, and when both look nearly ideal they stay with the left one — so the same sequence is chosen at different rates beside different partners, and people's left-choice rates and agreement with the majority differ in a pair-dependent way. Each sequence's impression is the current best judgement (personal ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on imbalance, the longest streak and complement symmetry, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_first_read_anchoring.py`

### longest_repeated_chunk_aversion — no comparison row

**Hypothesis:** People scan a sequence for its single most striking repeated chunk: the longest stretch of flips that appears again, unchanged, somewhere else in the same sequence (HTT in HTTHHTTH, HTHT in HTHTHTHT, HHT in THHTHHTT). One long recurring chunk makes the whole sequence look copied and so less random, by a shared weight on the length of that longest repeat (relative to the longest repeat the sequence's length allows). This repeat cue is added to the current best judgement (personal ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, personal weight on complement symmetry, and a personal guessing rate). It predicts that two sequences matching on switching, balance and streaks differ when one contains a long chunk that recurs and the other does not.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/longest_repeated_chunk_aversion.py`

### personal_indifference_band — no comparison row

**Hypothesis:** The change is in the decision rule, not in what makes a sequence look random. Each person has their own indifference band. When the two sequences seem about equally random to them, the difference does not register as a reason to pick either one, and they choose almost by coin flip. Only the part of the difference that goes beyond the band drives the choice, so a clear difference is followed as sharply as before. How wide the band is is a stable personal trait drawn from a population: people with a wide band agree with the majority only on clear-cut pairs, while people with a narrow band follow the consensus even on near-ties. This predicts that people differ in how consistently they agree with the majority more than guessing produces (the critique's excess spread of majority agreement), and that these differences sit on the close pairs. Each sequence's apparent randomness is judged exactly as in the current best model: a personal ideal switching rate and sensitivity, lenient toward over-alternation; personal weights on imbalance, the longest streak and complement symmetry; shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch; and a personal guessing rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_indifference_band.py`

### personal_pattern_detection_gain — no comparison row

**Hypothesis:** Refinement of the incumbent `personal_complement_symmetry_aversion`: each sequence is still judged by its switching rate against a personal ideal (personal sensitivity, lenient toward over-alternation), personal weights on heads/tails imbalance and the longest streak, the structural cues of a designed sequence (the final streak, mirror symmetry, complement symmetry, run-length variety and the most lopsided stretch), and a personal guessing rate. The one change is that noticing structure is a single personal trait rather than a trait specific to antisymmetry: each person has their own pattern-detection gain, drawn from a population, that scales all the structural cues together (now shared in their relative weights), so a person who spots palindromes also spots antisymmetry, rhythmic runs and local excesses, while another sees none of them — predicting that people's agreement with the majority is a stable personal trait that varies more across people than cue-specific weights produce (the critique's excess spread and split-half stability of majority agreement).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_pattern_detection_gain.py`

## Pruned models (out of the set; narrowest margin first)

### length_load_encoding_noise — pruned (experiment2 end of experiment)

**Margin:** 44.8 nats behind most_lopsided_stretch_aversion (4.1× dse)

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror` by a working-memory load process: to compare two sequences people must hold both in memory, and every additional flip adds encoding noise, so the same difference in how random the two look decides the choice sharply for short sequences and only weakly for long ones (the noise grows as a power of the number of flips, by a shared fitted exponent that could also come out negative if longer sequences give more evidence). How random each sequence looks is still the incumbent's judgement — each person's own ideal switching rate and sensitivity to it, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak and mirror symmetry — and people still guess on some trials at a personal lapse rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/length_load_encoding_noise.py`

### individual_alternation_sensitivity_mirror — pruned (experiment2 end of experiment)

**Margin:** 45.7 nats behind most_lopsided_stretch_aversion (4.1× dse)

**Hypothesis:** Refinement of the incumbent `mirror_symmetry_streak_aversion_lapse`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry (palindromes look designed) by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that how sharply a departure from one's ideal switching rate counts against a sequence is each person's own trait (drawn from a population) rather than shared: some people are strict, consistent judges of alternation and others barely discriminate — addressing the critique that people differ in how consistently they agree with the majority choice more than the incumbent's person-level weights and lapse rates produce.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/individual_alternation_sensitivity_mirror.py`

### asymmetric_ideal_rate_decisiveness — pruned (experiment2 end of experiment)

**Margin:** 55.8 nats behind most_lopsided_stretch_aversion (4.0× dse)

**Hypothesis:** Refinement of `personal_decisiveness_mirror_streak_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, still acts on their impression with a personal decisiveness, and still guesses on some trials at a personal lapse rate. The one change is that departures from one's ideal switching rate are judged asymmetrically: switching too often (over-alternation) counts against a sequence by a different, shared fraction of what switching too rarely (streakiness) does, rather than both directions counting equally — addressing the critique (surviving FDR) that among highly alternating pairs people choose the more alternating sequence far more often than a symmetric ideal-rate penalty implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/asymmetric_ideal_rate_decisiveness.py`

### individual_balance_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 58.0 nats behind individual_streak_aversion_lapse (4.5× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random, but people differ in how much that imbalance matters to them — each person has their own weight on H/T balance, drawn from a population distribution, instead of everyone sharing one. The one change is making the balance penalty individual, because the critique shows participants differ in their preference for the more balanced sequence far more than a single shared balance weight allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/individual_balance_ideal_alternation.py`

### personal_sensitivity_opening_run_lapse — pruned (experiment2 end of experiment)

**Margin:** 58.1 nats behind most_lopsided_stretch_aversion (4.5× dse)

**Hypothesis:** Refinement of `opening_run_streak_aversion_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the closing run and the opening run of identical flips by shared weights, and still guesses on some trials at a personal lapse rate. The one change is that how sharply a person's choices follow the distance from their ideal switching rate is their own trait (a personal alternation sensitivity drawn from a population, instead of one shared sensitivity), so some engaged people decide almost deterministically by alternation while others barely discriminate — addressing the critique that people differ in how consistently they agree with the majority choice more than the model's person-level ideals, weights and lapse rates produce.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/personal_sensitivity_opening_run_lapse.py`

### linear_distance_ideal_alternation_mirror — pruned (experiment2 end of experiment)

**Margin:** 61.4 nats behind most_lopsided_stretch_aversion (5.1× dse)

**Hypothesis:** Refinement of the incumbent `individual_alternation_sensitivity_mirror`: each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate (with their own sensitivity), still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak and mirror symmetry by shared weights, and still guesses on some trials at a personal lapse rate. The one change is the shape of the ideal-rate penalty: people count how many switches a sequence is off from their ideal and each missing or surplus switch costs the same amount (a penalty proportional to the distance from the ideal, not to its square), so small departures near the ideal are noticed and acted on rather than shrugged off, and large departures are not punished disproportionately — addressing the critique (surviving FDR) that among two highly alternating sequences people choose the more alternating one more often than the incumbent's quadratic ideal-rate penalty implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/linear_distance_ideal_alternation_mirror.py`

### edge_salient_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 69.2 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge a sequence's randomness by how often it switches between heads and tails compared with their own personal ideal switching rate, but they do not weigh every transition equally: attention and memory favour the beginning and end of the sequence (a serial-position effect), so switches and repeats near the edges count more than those in the middle. This disagrees most sharply with the current best model on pairs matched in overall alternation rate and heads/tails balance that differ only in where their repeats sit — the best model predicts indifference, whereas this one predicts that a streak at the start or end of a sequence makes it look much less random than the same streak buried in the middle.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/edge_salient_ideal_alternation.py`

### serial_position_weighted_alternation — pruned (experiment1 end of experiment)

**Margin:** 69.4 nats behind individual_streak_aversion_lapse (4.9× dse)

**Hypothesis:** People do not attend evenly to a coin-flip sequence: like a list read in order, its beginning and end are more salient than its middle (a serial-position effect), so the flip-to-flip switches and repeats at the edges of a sequence weigh more in their impression of how often it switches. Each person compares this salience-weighted switching rate with their own ideal switching rate for a random coin and picks the sequence that comes closer, so a streak sitting at the start or end of a sequence makes it look less random than the same streak buried in the middle.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/serial_position_weighted_alternation.py`

### ideal_alternation_with_balance — pruned (experiment1 end of experiment)

**Margin:** 77.1 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people additionally expect a random coin to give roughly equal numbers of heads and tails, so a sequence whose H/T counts are lopsided looks less random regardless of how it alternates. The one change is this shared penalty on H/T imbalance, added because the critique shows people prefer the more balanced sequence among pairs matched on alternation rate, which the alternation-only model predicts as indifference.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/ideal_alternation_with_balance.py`

### windowed_glimpse_expectation — pruned (experiment1 end of experiment)

**Margin:** 77.2 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** People read a coin-flip sequence through a limited working-memory window, holding only the last few flips at a time, and compare how mixed each glimpse is (its balance of heads and tails) with how mixed they expect a glimpse of a random coin to be; a sequence looks random to the extent its glimpses match that expectation. The expectation comes from each person's own belief about how often a random coin switches sides, so some people expect near-perfect mixing and others streakier glimpses, and because the window spans more than two flips, lopsided heads/tails counts and long runs are judged beyond what the alternation rate alone shows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/windowed_glimpse_expectation.py`

### personal_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 77.5 nats behind individual_streak_aversion_lapse (5.0× dse)

**Hypothesis:** Each person carries their own ideal switching rate for a random coin (how often consecutive flips should differ), and judges a sequence as more random the closer its proportion of alternations lies to that personal ideal; these ideals differ between people, some expecting near-perfect alternation and others streakier sequences. The model disagrees most with the current best model on highly alternating sequences (e.g. HTHTHTHT versus a streaky sequence), where it predicts a population split — some people strongly prefer them, others strongly reject them — and on pairs that differ in balance but not in alternation rate, where it predicts indifference.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_ideal_alternation.py`

### ideal_alternation_streak_penalty — pruned (experiment1 end of experiment)

**Margin:** 78.0 nats behind individual_streak_aversion_lapse (5.1× dse)

**Hypothesis:** Refinement of the incumbent `personal_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, but people also notice the single longest streak of identical flips and treat a long streak as a sign of non-randomness, beyond what the overall alternation rate shows. The one change is a shared penalty on the length of the longest run (in flips), added because the critique shows that among irregular sequences people penalise long runs beyond what alternation rate captures, which the alternation-only model under-predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/ideal_alternation_streak_penalty.py`

### mirror_symmetry_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Margin:** 78.1 nats behind most_lopsided_stretch_aversion (6.1× dse)

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the final streak by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is that people also notice mirror symmetry: a sequence that reads the same forwards and backwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks deliberately designed and so less random, by a shared weight — addressing the critique (surviving FDR) that people choose the palindrome of a pair less often than the incumbent's alternation, balance and streak terms predict.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/mirror_symmetry_streak_aversion_lapse.py`

### length_scaled_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 78.2 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still look less random, but people treat a departure from their ideal switching rate as evidence that grows with the number of flips it is observed over — the same off-ideal alternation rate is shrugged off in a short sequence and taken seriously in a long one. The one change is making alternation sensitivity grow with sequence length (as a power of the number of transitions, fitted), because the critique shows short sequences are judged by alternation less, relative to long ones, than the length-independent incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/length_scaled_ideal_alternation.py`

### periodic_chunk_repetition_aversion — pruned (experiment2 end of experiment)

**Margin:** 79.7 nats behind most_lopsided_stretch_aversion (6.4× dse)

**Hypothesis:** People see a sequence that is built by repeating a short chunk of three or four flips (translational repetition, such as HTTHTTHT or HTTHHTTH) as designed rather than random, so it looks less random by a shared weight on top of the incumbent's judgement (each person's ideal switching rate, personal weights on heads/tails imbalance and on the longest streak, a shared weight on the final streak and on mirror symmetry, and a personal guessing rate). This disagrees most sharply with the current best model on pairs where one sequence repeats a short chunk while having a near-ideal switching rate, good balance and no long streak (e.g. HTHHTHHT or HTTHTTHT against a streakier but non-repeating sequence): the best model calls these sequences highly random, while this model predicts people reject them.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/periodic_chunk_repetition_aversion.py`

### iter1_candidate1 — pruned (experiment1 end of experiment)

**Margin:** 82.0 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent it is better explained by a fair coin than by the non-random generators one might suspect — a biased coin (which explains unbalanced heads/tails counts) or a sticky/switchy coin (which explains long runs or rigid alternation). The one distortion is that each person's mental model of a "fair" coin switches between heads and tails at their own personal rate rather than exactly half the time, so people disagree about alternation while all still penalise imbalance and runs that the alternative generators explain.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/iter1_candidate1.py`

### mirror_symmetry_terminal_streak_lapse — pruned (experiment2 end of experiment)

**Margin:** 82.7 nats behind most_lopsided_stretch_aversion (5.6× dse)

**Hypothesis:** Refinement of `terminal_discounted_streak_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its alternation rate lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the most striking streak (with the run still in progress at the end weighed by a shared factor) by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that people notice mirror symmetry: a sequence of four or more flips that reads the same backwards as forwards (a palindrome such as HTTTTTTH or HHHTTHHH) looks designed, and so less random, by a shared weight — addressing the critique that people choose the palindrome of a pair less often than the alternation, balance and streak terms predict.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/mirror_symmetry_terminal_streak_lapse.py`

### divisive_contrast_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 84.3 nats behind individual_streak_aversion_lapse (5.3× dse)

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the two sequences against each other by divisive contrast: how much less random one looks than the other is weighed relative to how non-random the pair looks overall. Each sequence's non-randomness is its departure from the person's own ideal switching rate plus its heads/tails imbalance, but a given difference decides the choice strongly when the partner is nearly ideal and only weakly when both sequences are clearly non-random, so the same sequence is judged differently beside a different partner.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/divisive_contrast_ideal_alternation.py`

### opening_run_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Margin:** 101.2 nats behind most_lopsided_stretch_aversion (6.2× dse)

**Hypothesis:** Refinement of the incumbent `terminal_run_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, still weighs the streak the sequence ends on by a shared weight, and still guesses on some trials at a personal lapse rate. The one change is primacy: the run of identical flips a sequence opens with (the first thing read, which sets the first impression) also shifts its apparent randomness by a shared weight of its own, separate from the closing run's weight — addressing the critique (surviving FDR) that people choose the sequence with the longer opening run less often than the incumbent, which weighs the final run but not the first, predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/opening_run_streak_aversion_lapse.py`

### motif_stack_alternation_individual — pruned (experiment1 end of experiment)

**Margin:** 112.4 nats behind individual_streak_aversion_lapse (4.7× dse)

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the log-likelihood ratio of a fair coin versus Griffiths et al.'s four-motif stack automaton (the automaton held at the values the motif_stack fit settled on for this data), but people differ in how much they additionally favour or disfavour alternation — each participant carries their own preference for the sequence that switches between H and T more often, drawn from a population distribution. The single change is this participant-level alternation-preference random effect, added because the critique shows the between-participant spread in choosing the more-alternating sequence (SD 0.22) is four times what a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack_alternation_individual.py`

### individual_alternation_prototype — pruned (experiment1 end of experiment)

**Margin:** 117.7 nats behind individual_streak_aversion_lapse (6.0× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular, i.e. close to a prototype alternation rate and not periodic). The one change: each person holds their own prototype alternation rate — some expect a random sequence to switch much more often than a fair coin does, others barely more or even less — drawn from a population distribution, instead of everyone sharing one prototype. This addresses the critique that people differ far more in how strongly they prefer the more-alternating sequence than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/individual_alternation_prototype.py`

### max_lopsided_stretch_ideal — pruned (experiment1 end of experiment)

**Margin:** 118.7 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge a coin-flip sequence by its single most lopsided stretch: they notice the one contiguous run of flips in which heads most outnumber tails (or tails most outnumber heads), and a sequence looks random to the extent that this worst local excess matches what each person expects a random coin to produce. Each person carries their own ideal size for this worst stretch (relative to the sequence's length) — some expect even the most uneven stretch to be tiny, others tolerate a sizeable one — and the rest of the sequence, such as its overall alternation rate, matters only through that one most striking stretch.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/max_lopsided_stretch_ideal.py`

### local_window_ideal_alternation_lapse — pruned (experiment1 end of experiment)

**Margin:** 119.8 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** Refinement of the incumbent `personal_lapse_ideal_alternation`: people still judge a sequence as more random the closer its switching between heads and tails lies to their own personal ideal switching rate, and still guess on some trials at a personal, trait-like lapse rate, but they apply their ideal locally rather than to the sequence as a whole — they read the sequence a few flips at a time (a sliding stretch of three consecutive transitions) and each stretch is expected to switch at about the ideal rate, so a sequence's non-randomness is its average local departure from the ideal. The one change is this local (windowed) application of the ideal instead of the global alternation rate: two sequences with the same overall alternation rate differ when one bunches its repeats into a long streak and its switches into a rigid alternating stretch, which is exactly the streak aversion among alternation-matched pairs that the critique shows the incumbent under-predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/local_window_ideal_alternation_lapse.py`

### run_hazard_switch_belief — pruned (experiment1 end of experiment)

**Margin:** 121.2 nats behind individual_streak_aversion_lapse (5.3× dse)

**Hypothesis:** Refinement of `personal_switch_belief` (each person judges as more random the sequence that is more probable under their own subjective model of a coin as a process that switches between heads and tails with a personal probability). The one change: the believed coin has the gambler's fallacy built in — its chance of switching grows with the length of the current run (a shared rate of growth), on top of each person's own baseline switch belief — so a long run is judged far less probable than its number of switches alone implies. This addresses the critique that people penalise long runs beyond what the alternation rate captures.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/run_hazard_switch_belief.py`

### terminal_run_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Margin:** 123.2 nats behind most_lopsided_stretch_aversion (5.7× dse)

**Hypothesis:** Refinement of the incumbent `individual_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that the streak a sequence ends on is weighed differently from streaks elsewhere: the run of identical flips at the end of a sequence (the last thing read) shifts its apparent randomness by a shared weight of its own, which may be lenient or harsh, addressing the critique that people choose the sequence with the longer final run more often than the incumbent's position-blind streak term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/terminal_run_streak_aversion_lapse.py`

### terminal_discounted_streak_lapse — pruned (experiment2 end of experiment)

**Margin:** 125.9 nats behind most_lopsided_stretch_aversion (6.6× dse)

**Hypothesis:** Refinement of the incumbent `individual_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that the streak a sequence ends on is weighed differently from streaks that have already been closed off: because sequences are read left to right, a run still in progress at the last flip is judged as an unfinished streak, so its length counts by a fitted shared factor (discounted or amplified) when deciding which streak is the sequence's most striking one — addressing the critique that people choose the sequence with the longer final run more often than the position-blind longest-run term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/terminal_discounted_streak_lapse.py`

### personal_ideal_longest_run — pruned (experiment1 end of experiment)

**Margin:** 133.1 nats behind individual_streak_aversion_lapse (5.0× dse)

**Hypothesis:** People judge randomness by the longest streak a sequence contains: each person carries their own ideal length for the longest run of identical flips in a random sequence (relative to its length), and picks the sequence whose longest streak is closer to that personal ideal. Some people expect random sequences to contain almost no streaks, others expect a noticeable streak, so the same pair can be judged in opposite directions; the rest of the sequence's structure (its overall alternation rate or H/T balance) plays no role beyond what it does to the longest streak.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_ideal_longest_run.py`

### individual_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Margin:** 135.1 nats behind most_lopsided_stretch_aversion (5.3× dse)

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that people also notice the longest streak of identical flips and treat it as a sign of non-randomness by a weight that is each person's own (drawn from a population distribution, so some people are strongly streak-averse and others barely care), addressing the critique that individuals differ in preferring the sequence with the shorter longest run more than the incumbent produces, and that streaks are penalised beyond its alternation and balance terms.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/individual_streak_aversion_lapse.py`

### side_bias_streak_aversion_lapse — pruned (experiment2 end of experiment)

**Margin:** 135.6 nats behind most_lopsided_stretch_aversion (5.3× dse)

**Hypothesis:** People judge each sequence's randomness from how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is (each by weights of their own), and they guess on some trials at a personal lapse rate. The change is in the decision rule: each person also has a habitual leaning toward one response button (left or right), a position bias drawn from a population that may lean left overall, which tips every comparison toward that side by a fixed amount — so the bias matters most when the two sequences look about equally random, and a person's overall left-choice rate departs from one half regardless of which sequences they see.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/side_bias_streak_aversion_lapse.py`

### individual_irregularity_weight — pruned (experiment1 end of experiment)

**Margin:** 148.5 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/individual_irregularity_weight.py`

### edge_streak_balance_alternation_lapse — pruned (experiment2 end of experiment)

**Margin:** 154.2 nats behind most_lopsided_stretch_aversion (5.8× dse)

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that a streak sitting at the very start or end of a sequence is especially salient (primacy/recency of what is read first and last): the number of flips by which a run of identical flips at either edge exceeds two makes the sequence look less random by a shared weight, while the same streak buried in the middle carries no extra penalty — addressing the critique that among alternation-matched pairs people avoid the sequence with an edge streak more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/edge_streak_balance_alternation_lapse.py`

### personal_switch_belief — pruned (experiment1 end of experiment)

**Margin:** 159.2 nats behind individual_streak_aversion_lapse (6.0× dse)

**Hypothesis:** Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_switch_belief.py`

### primacy_recency_streak_balance_lapse — pruned (experiment2 end of experiment)

**Margin:** 184.0 nats behind most_lopsided_stretch_aversion (7.0× dse)

**Hypothesis:** Refinement of `edge_streak_balance_alternation_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate, and streaks at the edges of a sequence still carry an extra penalty. The one change is that the opening streak and the closing streak are no longer treated as one interchangeable "edge" streak: the streak a person reads last (recency) and the streak they read first (primacy) each make the sequence look less random by a weight of its own, so a run at the end can count for more than the same run at the start — addressing the critique that people penalise a long final run more than the incumbent's position-blind longest-run term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/primacy_recency_streak_balance_lapse.py`

### tally_drift_ideal — pruned (experiment1 end of experiment)

**Margin:** 186.0 nats behind individual_streak_aversion_lapse (5.9× dse)

**Hypothesis:** People keep a running tally of heads minus tails as they read a sequence, and judge it random to the extent that this tally wanders away from balance by about as much as they expect a fair coin's tally to wander: the whole path matters, so a streak that pushes the tally far off balance (even one later corrected) counts against a sequence, and so does a tally that never leaves balance at all. Each person has their own ideal amount of wandering — some expect the tally to hug zero, others tolerate large drifts — and picks the sequence whose tally drift is closer to that personal ideal.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/tally_drift_ideal.py`

### pattern_coverage_detection — pruned (experiment1 end of experiment)

**Margin:** 194.8 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** People judge randomness by pattern detection: they scan a sequence for perceptible regularities — a streak of identical flips or a stretch of strict H/T alternation — and a sequence looks non-random in proportion to how much of it is covered by such a detected pattern. A stretch is noticed as a pattern once it is long enough, with streaks noticed at a shared length and alternating stretches at a length that differs from person to person, so the whole shape of the sequence (how many of its flips sit inside long runs or long alternations, not its overall switch rate) drives the choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/pattern_coverage_detection.py`

### lapse_individual_streak_aversion — pruned (experiment2 end of experiment)

**Margin:** 195.6 nats behind most_lopsided_stretch_aversion (6.9× dse)

**Hypothesis:** Refinement of `personal_lapse_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guesses on some trials at a personal, trait-like lapse rate, but people also treat a long streak of identical flips as a sign of non-randomness, and how strongly a streak puts them off is each person's own trait (a personal weight on the longest run relative to sequence length, drawn from a population). The one change is this individual streak-aversion weight, added because the critique shows participants differ in their preference for the sequence with the shorter longest run more than the model's alternation, balance and lapse heterogeneity produces, and that streaks are penalised beyond alternation among alternation-matched pairs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/lapse_individual_streak_aversion.py`

### lapse_individual_balance_alternation — pruned (experiment2 end of experiment)

**Margin:** 211.4 nats behind most_lopsided_stretch_aversion (6.7× dse)

**Hypothesis:** Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/lapse_individual_balance_alternation.py`

### shared_stretch_cancellation — pruned (experiment2 end of experiment)

**Margin:** 277.2 nats behind most_lopsided_stretch_aversion (6.8× dse)

**Hypothesis:** People compare the two sequences side by side and cancel what they share: flips that both sequences have in common at their start and at their end are discounted as uninformative, and each sequence's randomness is judged only on the stretch where the two differ (plus the flip on either side of it, so its switches into and out of that stretch count). On that differing stretch each person still judges randomness by how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is, and guesses on some trials; so the same sequence is judged differently beside a partner that shares its opening or its ending than beside one that shares nothing.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/shared_stretch_cancellation.py`

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Margin:** 287.9 nats behind individual_streak_aversion_lapse (10.4× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously designed coin sequences — a solid streak (HHHH…), strict alternation (HTHT…), pairs (HHTT…), triples (HHHTTT…) and two halves (HHHH TTTT) — and a sequence looks non-random to the extent that it closely resembles (differs in few flips from) any of these remembered patterns, with similarity falling off exponentially with the share of mismatched flips. People differ in whether their remembered set of "designed" sequences prominently includes strict alternation, so some find alternating sequences contrived and others find them random-looking, while streak-like resemblance is penalised by everyone.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/exemplar_designed_pattern_similarity.py`

### worst_stretch_switch_deficit — pruned (experiment2 end of experiment)

**Margin:** 293.2 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge a coin-flip sequence by its single most un-random-looking stretch: they scan every contiguous stretch of flips and count how far its number of heads/tails switches falls short of (a streak) or exceeds (rigid alternation) the number they expect from their own ideal switching rate, and the sequence is only as random as its worst stretch. Because the shortfall is counted in switches rather than as a rate, a long stretch that is modestly off can outweigh a short one that is badly off, so a sequence's randomness is decided by one striking stretch rather than by its overall alternation rate, balance or longest run (people also guess on some trials at a personal rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/worst_stretch_switch_deficit.py`

### run_rhythm_evidence_personal_coin — pruned (experiment2 end of experiment)

**Margin:** 332.1 nats behind most_lopsided_stretch_aversion (8.9× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent that a coin explains it better than a "regular-rhythm" generator that builds sequences from runs of a few recurring lengths (so sequences whose runs all have the same length, such as HTHTHTHT, HHTTHHTT or one solid streak, look designed, and sequences with a variety of run lengths look random). The one distortion is that each person's mental model of the fair coin switches between heads and tails at their own personal rate rather than exactly half the time, so people differ in how much alternation they expect from a random coin while all still prize irregular run structure.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/run_rhythm_evidence_personal_coin.py`

### edge_weighted_alternation_memory — pruned (experiment2 end of experiment)

**Margin:** 350.6 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People encode a coin-flip sequence with serial-position effects in memory: the flips at the start and end of the sequence (primacy and recency) are encoded more strongly than those in the middle, so the switching pattern they perceive is dominated by what happens at the sequence's edges. Each person compares this edge-weighted impression of how often the sequence switches between heads and tails with their own personal ideal switching rate, and on some trials guesses at a personal lapse rate; a streak or rigid alternation sitting at an edge therefore makes a sequence look much less random than the same pattern buried in the middle.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/edge_weighted_alternation_memory.py`

### short_pattern_evenness_lapse — pruned (experiment2 end of experiment)

**Margin:** 446.4 nats behind most_lopsided_stretch_aversion (7.2× dse)

**Hypothesis:** People judge randomness by pattern evenness: they expect a random coin to produce every short pattern about equally often — heads and tails, each of the four flip pairs (HH, HT, TH, TT) and each of the flip triples — so a sequence looks non-random to the extent that a few short patterns dominate it (as in a streak, which is all HH, or strict alternation, which is all HT and TH), measured as how far the spread of its overlapping one-, two- and three-flip patterns falls short of the most even spread its length allows. How strongly this pattern unevenness puts a person off is their own trait, drawn from a population, and on some trials people guess at a personal lapse rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/short_pattern_evenness_lapse.py`

### local_representativeness — pruned (experiment1 end of experiment)

**Margin:** 467.0 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/local_representativeness.py`

### running_lead_surprise_path — pruned (experiment2 end of experiment)

**Margin:** 467.7 nats behind most_lopsided_stretch_aversion; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People keep a running tally of heads minus tails as they read a sequence left to right, and at every flip they ask how surprising the current lead is for a fair coin given how many flips they have seen so far (a lead of two after two flips is far more striking than a lead of two after eight). A sequence's impression of randomness is the average of this moment-by-moment surprise along the whole path, compared with each person's own ideal level of surprise (some expect the tally to hug balance, others expect it to wander), so an opening streak — which makes the early tally lopsided when few flips have been seen — counts against a sequence far more than the same streak at the end; on some trials a person guesses at a personal lapse rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/running_lead_surprise_path.py`

### motif_stack — pruned (experiment1 end of experiment)

**Margin:** 485.2 nats behind individual_streak_aversion_lapse (11.9× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack.py`

### exemplar_random_vs_designed_gcm — pruned (experiment2 end of experiment)

**Margin:** 531.0 nats behind most_lopsided_stretch_aversion (9.0× dse)

**Hypothesis:** People judge randomness by exemplar categorisation: they hold two remembered sets of same-length coin sequences, "random-looking" exemplars (balanced head/tail counts with no streak longer than two) and "designed" exemplars (a solid streak, strict alternation, repeated pairs HHTT…, repeated triples HHHTTT…, and two halves HHHHTTTT), and a sequence looks random to the extent its summed similarity to the random exemplars outweighs its summed similarity to the designed ones, with similarity falling off exponentially with the share of flips that differ. The sequence with the stronger random-versus-designed evidence is chosen, by a decisiveness that differs from person to person, and people guess on some trials at a personal lapse rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/exemplar_random_vs_designed_gcm.py`

### online_transition_surprise — pruned (experiment1 end of experiment)

**Margin:** 624.4 nats behind individual_streak_aversion_lapse (12.3× dse)

**Hypothesis:** People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/online_transition_surprise.py`

### lempel_ziv_incompressibility — pruned (experiment2 end of experiment)

**Margin:** 637.7 nats behind most_lopsided_stretch_aversion (8.2× dse)

**Hypothesis:** People judge a coin-flip sequence's randomness by how hard it is to describe compactly: reading it left to right, they notice whenever the next stretch of flips merely copies something already seen, and a sequence looks random in proportion to how many genuinely new chunks it takes to spell it out (its Lempel-Ziv compression complexity). How strongly this incompressibility drives the choice is each person's own trait, drawn from a population, so streaks, rigid alternation and repeated motifs all count against a sequence only through the single fact that they make it compressible.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/lempel_ziv_incompressibility.py`

### falk_konold_dp — pruned (experiment1 end of experiment)

**Margin:** 648.0 nats behind individual_streak_aversion_lapse (14.5× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/falk_konold_dp.py`

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Margin:** 816.9 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/finite_experience_occurrence.py`

### gist_typicality_class_size — pruned (experiment2 end of experiment)

**Margin:** 894.9 nats behind most_lopsided_stretch_aversion (12.0× dse)

**Hypothesis:** People judge randomness by typicality of a sequence's gist: they register only two summary facts — how many heads it has and how many runs (stretches of identical flips) it breaks into — and a sequence looks random to the extent that many other sequences of the same length share that same gist, so a gist that a fair coin could produce in many ways (moderate balance, moderate switching) looks random while a rare gist (one solid streak, strict alternation, all heads then all tails) looks designed. The sequence whose gist is more common is chosen, with a decisiveness that differs from person to person, and people guess on some trials at a personal lapse rate.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/gist_typicality_class_size.py`

### leaky_memory_bayesian_randomness — pruned (experiment2 end of experiment)

**Margin:** 948.4 nats behind most_lopsided_stretch_aversion (10.5× dse)

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent a fair coin explains it better than the non-random generators one might suspect (a biased coin, or a sticky/switchy coin whose switching probability is unknown), with the marginal likelihood of each alternative computed from the sequence's head/tail and repeat/switch counts. The one distortion is leaky memory: as they read left to right, earlier flips fade, so the evidence each flip contributes is discounted geometrically with its distance from the end of the sequence, which makes a streak or rigid pattern at the end of a sequence count as much stronger evidence of non-randomness than the same pattern at its start (people also guess on some trials at a personal rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/pruned/leaky_memory_bayesian_randomness.py`
