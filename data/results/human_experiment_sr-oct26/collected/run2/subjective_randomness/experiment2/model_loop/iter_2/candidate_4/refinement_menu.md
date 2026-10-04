# Refinement menu

The models you may refine, other than the incumbent `tally_span_personal_ideal`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### iter1_candidate4 — rank 1, 7.7 ± 37.1 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -2377.2

**Hypothesis:** Refinement of the incumbent `balance_aware_heads_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences), still favour heads-leaning sequences and still find lopsided H/T counts less random — but they also register the variety of short local patterns, so a sequence whose overlapping three-flip chunks are mostly different from one another (rather than the same few chunks recurring) looks more random, even when its overall switching rate is the same. The one change is a shared reward on the share of distinct three-flip chunks a sequence contains, addressing the critique (surviving FDR, q = 0.008) that among pairs with equal switch counts people pick the sequence with more distinct triplets far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/iter1_candidate4.py`

### streak_aware_balanced_alternation_ideal — rank 2, 8.6 ± 36.8 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -2378.2

**Hypothesis:** Refinement of the incumbent `balance_aware_heads_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences), still favour heads-leaning sequences and still find lopsided H/T counts non-random — but they also notice a streak of identical outcomes as a sign of non-randomness in its own right, so every flip of the longest run beyond two (HHH, HHHH, ...) makes a sequence look less random even when its overall switching rate is the same. The one change is a shared penalty on this absolute excess length of the longest streak, addressing the critique (surviving FDR, q = 0.008) that among pairs with equal switch counts people pick the sequence with the shorter longest run far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/streak_aware_balanced_alternation_ideal.py`

### balance_aware_heads_alternation_ideal — rank 3, 13.5 ± 38.3 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -2383.0

**Hypothesis:** Refinement of the incumbent `heads_default_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences) and still favour heads-leaning sequences, but they also expect a random coin to come out roughly half heads and half tails, so a sequence whose heads and tails counts are lopsided in either direction looks less random. The one change is a shared symmetric penalty on the H/T imbalance (|#H − #T| as a share of the length), addressing the critique that among pairs with equal switch counts people choose the more balanced sequence far more often than the incumbent's one-directional heads-share term allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/balance_aware_heads_alternation_ideal.py`

### balanced_heads_default_alternation_ideal — rank 4, 14.5 ± 37.7 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -2384.0

**Hypothesis:** Refinement of the incumbent `heads_default_alternation_ideal`: each person still judges a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences) and still treats heads as the default face, so heads-leaning sequences look more random — but people also expect a fair coin to produce roughly equal numbers of heads and tails, so a lopsided sequence in either direction looks less random. The one change is a shared symmetric penalty on the squared departure of the heads share from one half, addressing the critique (surviving FDR) that among pairs with equal switch counts people pick the more balanced sequence far more often than a model with only a linear heads-share term allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/balanced_heads_default_alternation_ideal.py`

### iter0_candidate0 — rank 5, 16.2 ± 34.0 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -2385.7

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent that it is better explained by a random coin than by a "rigged" process — either a coin biased toward one face (of unknown bias) or a sticky/switchy coin that tends to repeat or alternate (of unknown tendency) — so lopsided H/T counts and extreme switching both count as evidence of a non-random generator. The one distortion is in each person's picture of the random coin itself: rather than a fair, memoryless coin, each person believes a random coin switches sides at their own personal rate (usually more than half the time), and they choose the sequence with the higher posterior odds of having come from that subjective random coin.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/iter0_candidate0.py`

### heads_default_alternation_ideal — rank 6, 20.6 ± 41.2 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -2390.1

**Hypothesis:** People do not treat the two faces of the coin symmetrically: heads is the default, expected outcome of a coin toss, so a sequence dominated by tails reads as a coin that is "off" (biased toward the unusual face) and looks less random, while a heads-leaning sequence looks like ordinary coin flipping. This label asymmetry operates on top of each person's judgement of how close a sequence's switching rate is to their own ideal (with person-specific, length-scaled sensitivity and a shared penalty for visibly periodic sequences): of two sequences that switch equally often, people pick the one with more heads.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/heads_default_alternation_ideal.py`

### bayes_rigged_coin_with_two_flip_memory — rank 7, 28.7 ± 34.3 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -2398.2

**Hypothesis:** Refinement of `iter0_candidate0` (Bayesian detection of a rigged coin with a personally distorted random coin): people still judge a sequence as random by how much better a random coin (one that switches sides at their own personal rate) explains it than a rigged process does, and they still consider a coin biased toward one face and a sticky/switchy coin. The one change is a third rigged process they also entertain: a coin with a two-flip memory, whose next flip depends on the previous two (an unknown rule per two-flip context, such as "after HH comes H"). Sequences that reuse the same few triplets — long streaks, and short repeating patterns — are well explained by that memory coin and so look rigged even when their switch count matches the other sequence's, addressing the critique that at equal switch counts people pick the sequence with the shorter longest run and more distinct triplets.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/bayes_rigged_coin_with_two_flip_memory.py`

### balanced_heads_lapse_ideal_streak — rank 8, 43.7 ± 38.2 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -2413.2

**Hypothesis:** Refinement of `heads_favoring_lapse_ideal_streak`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate, still penalises every flip of the longest streak beyond two, still favours heads-leaning sequences, and still lapses to a random pick at their own rate — but people also expect a fair coin to produce roughly equal numbers of heads and tails, so a lopsided H/T count (in either direction) looks less random. The one change is a shared penalty on the absolute heads–tails imbalance of each sequence, addressing the critique's strongest discrepancy: among pairs with equal switch counts, people choose the more balanced sequence far more often than a model with only a directional heads-share term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/balanced_heads_lapse_ideal_streak.py`

### heads_favoring_lapse_ideal_streak — rank 9, 48.9 ± 38.8 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -2418.5

**Hypothesis:** Refinement of `personal_lapse_ideal_streak_excess`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, still penalises every flip of the longest streak beyond two, and still lapses to a random pick at their own rate — but the two coin faces are not psychologically symmetric: heads is the canonical, expected outcome of a coin toss, so a sequence with a larger share of heads reads as more like "real" coin flipping and looks more random. The one change is a shared heads-share bias inside the engaged decision, addressing the critique that among pairs with equal switch counts people choose the heads-heavier sequence more often than an H/T-symmetric model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/heads_favoring_lapse_ideal_streak.py`

### switching_then_balance_semiorder — rank 10, 63.8 ± 34.9 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -2433.3

**Hypothesis:** People compare the two sequences lexicographically (a semiorder): they first ask which one switches between H and T at a rate closer to their own personal ideal for a random coin; only when the two sequences switch about equally often does a second comparison take over, and then they pick the one whose heads and tails are more evenly balanced. Because the balance cue is consulted only when the partner leaves the switching comparison undecided, the same sequence's balance decides the choice beside a partner with the same number of switches and is ignored beside a partner that switches clearly more or less.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/switching_then_balance_semiorder.py`

### personal_side_bias_alternation_ideal — rank 11, 110.3 ± 37.1 nats behind the best (3.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2479.9

**Hypothesis:** The decision rule, not the evidence, carries a stable individual trait: each person has their own habitual lean toward clicking the left or the right button, which adds to their impression of which sequence is more random on every trial (people differ in the direction and strength of this response-side bias, and it is unrelated to the sequences shown). The impression itself is kept simple: a sequence looks random to the extent that its proportion of H/T switches is close to the person's own ideal switching rate, so the side bias decides near-ties and shifts close calls toward the favoured side.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/personal_side_bias_alternation_ideal.py`

### max_lopsided_window_surprise — rank 12, 264.8 ± 56.3 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2634.3

**Hypothesis:** People judge a sequence by its single most lopsided stretch: they scan every contiguous stretch of flips and register the one whose heads/tails split is most improbable for a fair coin of that stretch's length (a streak, or a stretch like HHTHHH that is nearly all one face), and that one most surprising stretch alone decides how non-random the sequence looks — the rest of the sequence does not count. They pick the sequence whose most lopsided stretch is less surprising, with people differing in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/max_lopsided_window_surprise.py`

### gist_typicality_count_runs — rank 13, 585.5 ± 81.8 nats behind the best (7.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2955.0

**Hypothesis:** People judge how random a sequence looks by how typical its gist is for a fair coin: they register only two coarse summaries of a sequence — how many heads it has and how many runs (streaks) it breaks into — and a sequence looks random to the extent that many coin sequences of that length share that same gist. Lopsided counts, long streaks (few runs) and strict alternation (maximal runs) are all rare gists and look non-random, while balanced sequences with a middling number of runs are common gists and look random; people differ only in how strongly this felt typicality drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/gist_typicality_count_runs.py`

## Pruned models (out of the set; narrowest margin first)

### periodic_ideal_absolute_streak_penalty — pruned (experiment1 end of experiment)

**Margin:** 26.2 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 14

**Hypothesis:** Refinement of the incumbent `periodic_penalized_alternation_ideal`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, with a shared penalty for visibly periodic sequences, but people also notice a streak of identical outcomes as soon as it is long in absolute terms — every flip beyond two in the longest run (HHH, HHHH, ...) counts against randomness, however long the sequence. The one change is this shared penalty on the absolute excess length of the longest streak (counted in flips, not as a share of the sequence), addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/periodic_ideal_absolute_streak_penalty.py`

### streak_penalized_personal_prototype — pruned (experiment1 end of experiment)

**Margin:** 32.4 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 14

**Hypothesis:** Refinement of `person_specific_alternation_prototype` (local representativeness with a person-specific preferred alternation rate): people judge randomness by multiscale local H/T balance and by irregularity — how far a sequence's switching rate is from their own preferred rate, plus a periodic-template penalty — and, as the one added component, they treat the longest streak in a sequence as a salient sign of non-randomness, penalising long runs beyond what the overall switching rate implies. (The distance from the personal preferred rate is taken as smooth and squared, as in the incumbent, rather than absolute; the claim is the added streak penalty.) This targets the critique that the best model under-penalises long streaks and regular patterns once alternation rate is accounted for.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/streak_penalized_personal_prototype.py`

### length_scaled_alternation_ideal_streak — pruned (experiment1 end of experiment)

**Margin:** 37.1 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 14

**Hypothesis:** Refinement of `personal_alternation_ideal_streak_penalty`: each person judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, with the longest streak counted as a further sign of non-randomness — but the alternation impression is treated as evidence that accumulates over the flips seen, so a given departure from the personal ideal weighs more heavily in a longer sequence (it rests on more transitions) than in a short one. The one change is this length-scaled sensitivity to the alternation-rate distance (sensitivity grows as a fitted power of the number of transitions), addressing the critique that people's preference for the higher-switch sequence falls off with length far less than a fixed-sensitivity, rate-based model predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/length_scaled_alternation_ideal_streak.py`

### personal_alternation_ideal_streak_penalty — pruned (experiment1 end of experiment)

**Margin:** 38.0 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 14

**Hypothesis:** Refinement of `personal_alternation_ideal`: each person still judges a sequence as random to the extent that its proportion of H/T switches is close to their own personal ideal switching rate, but people also treat the longest streak of identical outcomes as a salient sign of non-randomness in its own right. The one change is a shared penalty on the sequence's longest run (relative to its length), so that of two sequences equally close to a person's ideal switching rate, the one containing a longer streak looks less random — addressing the critique that long streaks are penalised beyond what the alternation rate explains.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_alternation_ideal_streak_penalty.py`

### personal_alternation_ideal — pruned (experiment1 end of experiment)

**Margin:** 42.1 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 14

**Hypothesis:** Each person carries their own ideal switching rate for a random coin — some expect a fair coin to flip side about half the time, others expect it to alternate far more often — and judges a sequence as random to the extent that its proportion of H/T switches is close to that personal ideal. The current best model assumes one shared alternation prototype for everyone; this model disagrees most sharply on pairs where both sequences alternate heavily (e.g. HTHTHTHT versus HTHHTHTH), which strong over-alternators should split decisively toward the more alternating sequence while moderate people choose the other.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_alternation_ideal.py`

### pair_normalized_alternation_contrast — pruned (experiment1 end of experiment)

**Margin:** 59.8 nats behind graded_periodicity_personal_ideal (4.6× dse)

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the pair relative to itself. Each person compares how far each sequence's switching rate is from their own ideal switching rate, but the difference between the two is weighed relative to how far both sequences are from that ideal together (divisive contrast normalisation, as in Weber's law): a given gap decides the choice sharply when both sequences are near the ideal and barely matters when both are far from it, so the same sequence is chosen more or less decisively depending on its partner.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/pair_normalized_alternation_contrast.py`

### local_window_alternation_ideal — pruned (experiment1 end of experiment)

**Margin:** 61.1 nats behind graded_periodicity_personal_ideal (4.8× dse)

**Hypothesis:** Refinement of the incumbent `length_scaled_alternation_ideal`: each person still judges a sequence as random by how close its switching to their own ideal switching rate is (sensitivity scaling as a power of the number of transitions, with a shared penalty for visibly periodic sequences), but they apply that ideal locally rather than to the sequence as a whole — they check every short stretch of three flips against their ideal and average how far each stretch is from it. The one change is this local reading of the alternation ideal: a sequence whose switches are bunched together, leaving a long streak with no switching, contains stretches far from the ideal and so looks less random than a sequence with the same total number of switches spread evenly, addressing the critique that among pairs with equal switch counts people prefer the sequence with the shorter longest run more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/local_window_alternation_ideal.py`

### person_specific_alternation_prototype — pruned (experiment1 end of experiment)

**Margin:** 81.5 nats behind graded_periodicity_personal_ideal (4.9× dse)

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky local representativeness): people judge randomness by local H/T balance across scales plus irregularity measured as distance from a preferred alternation rate and a periodic-template penalty, but each person holds their own preferred alternation rate (prototype) rather than one shared by everyone. The single change is making the alternation prototype person-specific, drawn from a population distribution, which addresses the critique that the model under-produces individual differences in preference for alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/person_specific_alternation_prototype.py`

### leaky_switch_impression_tracking — pruned (experiment1 end of experiment)

**Margin:** 108.1 nats behind graded_periodicity_personal_ideal (5.1× dse)

**Hypothesis:** People read a sequence flip by flip and keep a running impression of how often the coin is switching, updated after every flip by a leaky (exponentially forgetting) memory in which recent transitions weigh more than earlier ones; at every point of the reading they compare this running impression with their own ideal switching rate, and the sequence looks random to the extent that the impression stays close to that ideal throughout. Because the impression tracks the reading moment by moment, a long streak drags it far below the ideal for several flips (and a stretch of strict alternation pushes it above), so a sequence with bunched switches looks less random than one with the same number of switches spread evenly; people differ in their ideal switching rate, while the memory's forgetting rate is shared.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/leaky_switch_impression_tracking.py`

### motif_stack_person_alternation — pruned (experiment1 end of experiment)

**Margin:** 115.0 nats behind graded_periodicity_personal_ideal; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 14

**Hypothesis:** Refinement of `motif_stack`: people judge randomness as the likelihood ratio of a fair coin against Griffiths et al.'s four-motif stack automaton, a regularity detector shared by everyone (held at the automaton settings `motif_stack` itself estimated on these data), but individuals differ in how strongly they additionally treat frequent alternation itself as a sign of randomness. The one change is a person-specific alternation-preference weight, drawn from a population distribution, on the difference in alternation rate between the two sequences — addressing the critique that people differ far more in their preference for alternation than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack_person_alternation.py`

### person_alternation_prototype — pruned (experiment1 end of experiment)

**Margin:** 115.3 nats behind graded_periodicity_personal_ideal (6.4× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky local representativeness: a sequence looks random when it is locally balanced and irregular, irregularity being closeness to an over-alternating prototype plus a periodic-template penalty). The one change: each person carries their own ideal alternation rate (prototype), drawn from a population distribution, rather than everyone sharing a single prototype — so people differ in how much alternation they expect from a random coin, which the single-population incumbent cannot produce (the critique's large across-participant spread in preference for the more-alternating sequence).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/person_alternation_prototype.py`

### gamblers_fallacy_fading_memory — pruned (experiment1 end of experiment)

**Margin:** 136.3 nats behind graded_periodicity_personal_ideal (4.1× dse)

**Hypothesis:** People judge randomness by reading a sequence flip by flip with a gambler's-fallacy expectation: after each flip they expect the coin to "balance out" against the flips they have just seen, with recent flips weighing more than earlier ones (a fading memory), and a sequence looks random to the extent that its flips keep matching these balancing expectations (its probability under that self-correcting subjective coin). Because the expectation of a switch builds up the longer a streak continues, a long run is increasingly surprising beyond what its number of switches implies; people differ in how strongly they hold this balancing expectation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/gamblers_fallacy_fading_memory.py`

### bayes_pattern_vs_alternating_coin — pruned (experiment1 end of experiment)

**Margin:** 153.5 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent it is better explained by a random coin than by a noisy repeating pattern (a short template of period 1-4, such as HHHH, HTHT or HHTT, copied with occasional errors). The single distortion is in their picture of the random coin: each person believes a fair coin switches sides at their own rate (often more than half the time), so alternation counts as evidence for randomness up to the point where the sequence becomes regular enough to be better explained by a repeating template.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/bayes_pattern_vs_alternating_coin.py`

### personal_markov_coin_belief — pruned (experiment1 end of experiment)

**Margin:** 205.7 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_markov_coin_belief.py`

### running_tally_drift_ideal — pruned (experiment1 end of experiment)

**Margin:** 234.8 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails (or tails lead heads), and judge randomness by how far that tally wanders from balance along the way: each person expects a fair coin's tally to drift away from even by some typical amount (scaled to how many flips have been read so far), and a sequence looks random to the extent that its average drift matches that personal expectation. A long streak makes the tally run away from balance (too much drift), while strict alternation pins it at even (too little drift), so the path of the tally — not the final count or the switch rate — decides the choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/running_tally_drift_ideal.py`

### chunk_entropy_variety — pruned (experiment1 end of experiment)

**Margin:** 246.4 nats behind graded_periodicity_personal_ideal (4.1× dse)

**Hypothesis:** People judge randomness by the variety of short patterns a sequence contains: reading it as overlapping chunks of one to four flips, a sequence looks random to the extent that its chunks of each size are spread evenly over the different possible chunks (high chunk entropy for its length), and repetitive to the extent that the same few chunks recur — through a lopsided H/T count, a long streak, strict alternation, or an almost-repeating unit with one slip. People differ in how strongly this felt variety drives their choice. This disagrees most sharply with the switch-rate-ideal incumbent on pairs with similar switch counts where one sequence reuses its chunks (near-periodic sequences, or one with a long streak), and on block-structured sequences like HHTTHHTT, whose middling switch rate the incumbent rates as plausible but whose chunk variety is low.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/chunk_entropy_variety.py`

### most_salient_regular_stretch — pruned (experiment1 end of experiment)

**Margin:** 301.1 nats behind graded_periodicity_personal_ideal (9.6× dse)

**Hypothesis:** People judge a sequence by its single most striking regular stretch: they scan it for the longest streak of identical flips and the longest stretch of perfect alternation, and whichever of these looks most patterned (a smooth maximum of the two, not a sum over the whole sequence) decides how non-random the sequence seems; they choose the sequence whose most striking stretch is less salient. Everyone finds both kinds of stretch a sign of pattern, and people share how salient a streak is but differ in how salient a stretch of strict alternation is (for some it is a glaring pattern, for others it barely registers, so for them a long streak decides almost every choice).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/most_salient_regular_stretch.py`

### position_weighted_switch_impression — pruned (experiment1 end of experiment)

**Margin:** 312.2 nats behind graded_periodicity_personal_ideal (7.2× dse)

**Hypothesis:** People do not weigh every part of a sequence equally: they form their impression of how often the coin switches mainly from one end of the sequence (the last flips they read, or the first), and judge a sequence as more random the closer that position-weighted switch rate comes to a shared ideal switch rate for a random coin. Two sequences with the same overall number of switches can therefore be judged differently depending on whether their switches or their repeats sit at the attended end.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/position_weighted_switch_impression.py`

### local_representativeness — pruned (experiment1 end of experiment)

**Margin:** 334.3 nats behind graded_periodicity_personal_ideal (8.0× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/local_representativeness.py`

### motif_stack — pruned (experiment1 end of experiment)

**Margin:** 381.2 nats behind graded_periodicity_personal_ideal (8.2× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack.py`

### context_learner_surprise — pruned (experiment1 end of experiment)

**Margin:** 512.8 nats behind graded_periodicity_personal_ideal (7.7× dse)

**Hypothesis:** People read a sequence flip by flip and, with a short memory of the last two flips, keep trying to predict the next flip from what followed the same two-flip context earlier in the sequence (an online pattern learner); a sequence looks random to the extent that these running predictions keep failing, i.e. by the average surprise the learner experiences. Long streaks and repeating patterns (HTHTHTHT, HHTTHHTT) quickly become predictable to such a learner and so look non-random even when their overall switch rate is near what people expect, and people differ in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/context_learner_surprise.py`

### falk_konold_dp — pruned (experiment1 end of experiment)

**Margin:** 518.3 nats behind graded_periodicity_personal_ideal (11.3× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/falk_konold_dp.py`

### best_lag_copy_rule_detector — pruned (experiment1 end of experiment)

**Margin:** 610.5 nats behind graded_periodicity_personal_ideal (9.4× dse)

**Hypothesis:** People judge randomness by searching each sequence for a simple copy rule — "each flip repeats the flip k places back" for some small lag k (lag 1 catches streaks, lag 2 catches HTHT alternation, lag 4 catches HHTTHHTT) — and a sequence looks non-random to the extent that its best such rule predicts its flips. They choose the sequence whose best copy rule fits worse, with people differing only in how strongly that detected regularity drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/best_lag_copy_rule_detector.py`

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Margin:** 657.0 nats behind graded_periodicity_personal_ideal (9.7× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/finite_experience_occurrence.py`

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Margin:** 668.3 nats behind graded_periodicity_personal_ideal (10.6× dse)

**Hypothesis:** People judge randomness by exemplar memory: they hold remembered examples of "designed" coin sequences — streaks (HHHH), strict alternation (HTHT), and short repeating units (HHT, HHTT, HHHT, HTTT and their shifts and mirror images) — and a sequence looks non-random to the extent that it is similar to these stored patterns, with similarity falling off exponentially with the number of flips that would have to change to turn the sequence into a stored pattern (summed over all stored examples, as in a generalized context model). Random examples are remembered too diffusely to favour any particular sequence, so people pick the sequence with less summed similarity to the designed exemplars; individuals differ in how strongly the alternation exemplars are stored, so for some people near-alternating sequences look designed and for others they do not.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/exemplar_designed_pattern_similarity.py`
