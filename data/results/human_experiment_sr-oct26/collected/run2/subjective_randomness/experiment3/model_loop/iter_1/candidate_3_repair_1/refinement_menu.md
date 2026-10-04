# Refinement menu

The models you may refine, other than the incumbent `rule_built_tally_switch_triplet`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### iter0_candidate4 — rank 1, 10.3 ± 5.6 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -3423.9

**Hypothesis:** Refinement of the incumbent `tally_span_switch_ideal_periodic_unit`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, personal left/right lean), still judge switching by closeness to a shared ideal switching rate with a person-specific weight, and still see a sequence built by repeating a short unit (HTHT..., HHTHHT) as designed — but they also register the variety of short local patterns, so of two sequences that are both non-periodic, the one whose overlapping three-flip chunks are mostly different from one another looks more random than one that recycles the same few chunks (near-repeats with a slip, block structure, streaks). The one change is a shared reward on the share of distinct three-flip chunks (taken from `tally_span_switch_ideal_triplet_variety`), grading the all-or-none periodicity judgement so that near-patterned sequences are also penalised, addressing the critique that items are more polarised than the incumbent predicts (it misses stimulus features driving strong consensus).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/iter0_candidate4.py`

### graded_near_periodicity_tally_switch — rank 2, 18.2 ± 9.1 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -3431.9

**Hypothesis:** Refinement of the incumbent `tally_span_switch_ideal_periodic_unit`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, personal left/right lean), and still judge switching by closeness to a shared ideal switching rate with a person-specific weight, and they still see a sequence built by repeating a short unit as designed rather than random. The one change is that this sense of a repeating pattern is graded rather than all-or-nothing: people register how nearly the sequence repeats itself at its best-fitting period of two or more flips (the share of flips that match the flip one unit earlier), so a sequence that almost repeats a motif with one slip (HHTHHTHT, HTTHTTHH) also looks designed, and the impression grows sharply (squared) as the repetition approaches perfect. This targets the critique that people's choices on individual pairs are more polarised than the incumbent predicts, which its exact-periodicity switch cannot produce for near-periodic sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/graded_near_periodicity_tally_switch.py`

### tally_span_switch_ideal_periodic_unit — rank 3, 21.9 ± 9.2 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3435.5

**Hypothesis:** Refinement of the incumbent `tally_span_switch_rate_ideal_2`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, a personal left/right lean), and still judge switching by closeness to a shared ideal switching rate with a person-specific weight — but they also spot a short unit being repeated (strict alternation HTHT..., or motifs like HHTHHT and HTTTHTTT, any unit of two or more flips repeated at least twice through the whole sequence), and a sequence with such a visible repeating pattern looks designed rather than random. The one change is this shared penalty on visible periodicity (taken from the alternation-ideal models), which lets the switching ideal sit higher so that heavy switching is not over-penalised, addressing the critiques (surviving FDR) that people choose periodic period-3/4 motifs less often, and the more-switching sequence among high-switch pairs more often, than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_span_switch_ideal_periodic_unit.py`

### tally_span_switch_ideal_triplet_variety — rank 4, 23.6 ± 11.5 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3437.2

**Hypothesis:** Refinement of the incumbent `tally_span_switch_rate_ideal_2`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (person-specific, length-scaled sensitivity, personal left/right lean), and still judge switching by closeness to an ideal switching rate with a person-specific weight — but they also register the variety of short local patterns, so a sequence whose overlapping three-flip chunks are mostly different from one another looks more random than one that recycles the same few chunks (strict alternation HTHTHTHT uses only HTH and THT; repeating motifs such as HHTHHT use only three). The one change is a shared reward on the share of distinct three-flip chunks (taken from `tally_span_switch_triplet_variety`), which lets the switching ideal sit higher while rejecting regular high-switch sequences, addressing the critiques (surviving FDR) that people choose periodic-motif sequences less, the more-switching sequence among high-switch pairs more, and the sequence with more distinct triplets at equal switch counts more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_span_switch_ideal_triplet_variety.py`

### tally_span_switch_ideal_periodic_unit_2 — rank 5, 75.5 ± 15.1 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3489.1

**Hypothesis:** Refinement of `tally_span_switch_rate_ideal` (running heads-minus-tails tally judged by closeness of its span to a personal expected span, switching judged by closeness to a shared ideal rate with person-specific weight, personal left/right lean): people additionally notice when a sequence is visibly built by repeating a short unit — strict alternation HTHTHTHT, HHTHHT, HHTTHHTT, HTTTHTTT — and see such a repeating pattern as designed, not random, regardless of its span or switching rate. The one change is a shared penalty on sequences that are exactly periodic (a unit repeated at least twice), which lets the switching ideal stop doing the work of rejecting strict alternation, addressing the critiques (surviving FDR) that people choose periodic-motif sequences less often, and the more-switching sequence among high-switch pairs more often, than the current models predict.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_span_switch_ideal_periodic_unit_2.py`

### tally_span_switch_rate_ideal_2 — rank 6, 77.8 ± 18.0 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3491.5

**Hypothesis:** Refinement of the incumbent `iter2_candidate3`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span (with person-specific, length-scaled sensitivity and a personal left/right lean), and still attend to how often the coin switches sides with person-specific strength — but switching is no longer rewarded "the more the better": people expect a random coin to switch at a particular rate (somewhat above one half), so a sequence that switches less than that looks streaky and one that switches more, up to strict alternation like HTHTHTHT, looks too regular. The one change replaces the incumbent's linear switching reward with a closeness-to-an-ideal-switching-rate judgement (a shared ideal rate, person-specific weight), addressing the critiques (surviving FDR) that people pick a perfect alternator far less often than the linear reward implies and that the alternation preference changes shape among high-switching pairs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_span_switch_rate_ideal_2.py`

### pair_normalized_tally_switch_contrast — rank 7, 77.9 ± 18.2 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3491.5

**Hypothesis:** People judge the two sequences against each other, not one at a time: each person still compares each sequence's running heads-minus-tails tally span with their own expected span and its switching rate with an ideal switching rate (with a personal left/right lean), but the difference between the two sequences on each cue is weighed relative to how far the pair as a whole sits from the ideal (divisive contrast normalisation). A given gap decides the choice sharply when both sequences are near the ideal and only weakly when both are far from it, so the same sequence is chosen more or less decisively depending on its partner — for instance, between two heavily switching sequences people barely penalise the one that switches more, unlike when it is paired with a moderately switching sequence.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/pair_normalized_tally_switch_contrast.py`

### feature_space_random_designed_exemplars — rank 8, 435.2 ± 51.8 nats behind the best (8.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3848.8

**Hypothesis:** People judge randomness by exemplar memory in a simple perceptual space: they represent a sequence by how often it switches sides, how far its running heads-minus-tails tally strays from even, and whether it visibly repeats a unit, and they compare it with remembered examples — one remembered "typical random" sequence (whose tally swing differs from person to person) and a handful of remembered designed sequences of the same length (a streak HHHH..., strict alternation HTHT..., and the repeating units HHT, HHTT, HHHT). A sequence looks random to the extent that it is more similar to the random exemplar than to its nearest designed exemplars (similarity falling off as a Gaussian of attention-weighted distance), and people pick the sequence with the higher random-versus-designed similarity ratio, apart from a small personal left/right lean.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/feature_space_random_designed_exemplars.py`

### tally_restoring_force_coin — rank 9, 972.3 ± 103.3 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4386.0

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails, and expect a random coin to be self-correcting at the level of that tally: the further the tally has drifted from even, the more they expect the next flip to pull it back toward balance, and a flip that pushes the tally further away is more surprising the larger the current lead. A sequence looks random to the extent that its path is probable under this balance-restoring subjective coin (relative to a fair coin), so a flip that extends an already large lead counts strongly against randomness while the same flip at an even tally costs nothing; people choose the sequence with the higher path probability, differing in how strongly this drives their choice and with a small habitual left/right lean.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/tally_restoring_force_coin.py`

### shared_indifference_band_tally_switch — rank 10, 987.6 ± 64.9 nats behind the best (15.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4401.2; PSIS-LOO unreliable (3% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People do not choose in proportion to how much more random one sequence looks: their decision rule has an indifference band. When the felt difference in randomness between the two sequences is small, they are effectively indifferent and pick close to a coin flip; only the part of the difference that exceeds this band pushes them toward a side, so clearly different pairs are decided with strong consensus while near-ties stay near 50/50. The evidence is kept as simple as possible and shared by everyone (how close a sequence's running heads-minus-tails tally span is to an expected span, and how close its switching rate is to an ideal rate); the claim is the shared indifference band in the decision rule, which predicts more polarised item-level choice rates than a plain logistic choice rule.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/shared_indifference_band_tally_switch.py`

### shared_judgement_personal_engagement — no comparison row

**Hypothesis:** Everyone shares the same impression of how random a sequence looks (how close its running heads-minus-tails tally span and its switching rate are to what a fair coin should do, with a reward for varied three-flip chunks and a penalty for sequences built by a visible copying or mirror rule); what differs between people is engagement, not perception. Each person has their own rate of disengaged trials on which they do not evaluate the sequences at all and simply click their habitual side, so people differ in how often they agree with the consensus choice while sharing its direction and ordering — some people track the consensus almost perfectly and others are close to chance on every pair.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/shared_judgement_personal_engagement.py`

### personal_engagement_lapse_tally_switch — no comparison row

**Hypothesis:** People differ in how consistently they engage with the task at all: on each trial a person either actually compares the two sequences or, at a personal rate (their own disengagement trait, drawn from a population), skips the comparison and picks a side by guessing. When engaged, they judge randomness as the current best model describes (how close the running heads-minus-tails tally's span is to their own expected span, how close the switching rate is to an ideal rate, variety of three-flip chunks, a penalty on visibly rule-built sequences, and a left/right lean); the claim is the person-specific engagement rate in the decision stage, which makes some people agree with the majority on clear-cut pairs almost always and others barely above chance. It disagrees most sharply with the current best model on pairs with strong consensus (e.g. a rule-built or streaky sequence against a balanced irregular one), where it predicts a population of near-chance responders rather than uniformly extreme choice rates, addressing the critique that people differ in their agreement with the majority more than the best model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/personal_engagement_lapse_tally_switch.py`

### shared_impression_personal_decisiveness — no comparison row

**Hypothesis:** Everyone shares one impression of what makes a coin sequence look random (a running heads-minus-tails tally whose span is close to an expected span, switching close to an ideal rate, varied three-flip chunks, and no visible construction rule), but people differ in how consistently they act on that impression: each person has their own decisiveness, a single gain on the whole felt difference in randomness between the two sequences. Individual differences therefore lie in decision noise — some people pick the more random-looking sequence almost every time, others only weakly — rather than in what randomness looks like, which predicts a wider spread across people in how often they agree with the majority than cue-specific individual differences do.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment3/model_loop/models/shared_impression_personal_decisiveness.py`

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

### tally_span_switch_rate_ideal — pruned (experiment2 end of experiment)

**Margin:** 40.5 nats behind tally_span_switch_ideal_periodic_unit (4.1× dse)

**Hypothesis:** Refinement of `tally_span_ideal_plus_switch_reward`: people still keep a running heads-minus-tails tally and pick the sequence whose tally span is closer to their own expected span (with a personal left/right lean), and they still judge how often the coin switches sides in its own right — but switching is not rewarded without limit: people expect a random coin to switch at some ideal rate (above one half, below always), so more switching looks more random only up to that rate, and near-perfect alternation such as HTHTHTHT looks too regular. The one change replaces the model's linear switch reward with closeness to a shared ideal switching rate (an inverted U, with people differing in how much this closeness weighs), addressing the critique (surviving FDR) that people choose the perfect alternator, and the more-switching sequence among high-switch pairs, less often than a linear switch reward predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tally_span_switch_rate_ideal.py`

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

### tally_span_switch_triplet_variety — pruned (experiment2 end of experiment)

**Margin:** 81.6 nats behind tally_span_switch_ideal_periodic_unit (4.4× dse)

**Hypothesis:** Refinement of the incumbent `iter2_candidate3`: people still keep a running heads-minus-tails tally and judge a sequence random by how close the tally's span is to their own expected span, still reward switching with a person-specific weight, and still have a personal left/right lean — but they also register the variety of short local patterns, so a sequence whose overlapping three-flip chunks are mostly different from one another looks more random than one that recycles the same few chunks (as strict alternation HTHTHTHT does, using only HTH and THT). The one change is a shared reward on the share of distinct three-flip chunks (taken from `iter1_candidate4`), addressing the critiques (surviving FDR) that at equal switch counts people choose the sequence with more distinct triplets more than the incumbent predicts, and that they choose a perfect alternator less often than the incumbent's linear switching reward implies.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tally_span_switch_triplet_variety.py`

### iter2_candidate3 — pruned (experiment2 end of experiment)

**Margin:** 96.7 nats behind tally_span_switch_ideal_periodic_unit (5.0× dse)

**Hypothesis:** Refinement of the incumbent `tally_span_personal_ideal`: people still keep a running heads-minus-tails tally while reading a sequence and judge it random by how close the tally's span is to their own expected span (with a personal left/right lean), but they also register how often the coin switches sides as a separate sign of randomness, so of two sequences whose tallies swing equally widely they pick the one that alternates more — and people differ in how strongly they reward alternation. The one change is this person-specific preference for the share of H/T switches, addressing the critique (surviving FDR) that among pairs with near-equal tally spans people choose the more-switching sequence far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/iter2_candidate3.py`

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

### tally_span_ideal_plus_switch_reward — pruned (experiment2 end of experiment)

**Margin:** 146.3 nats behind tally_span_switch_ideal_periodic_unit (5.5× dse)

**Hypothesis:** Refinement of the incumbent `tally_span_personal_ideal`: people still read a sequence flip by flip, keep a running tally of how far heads lead tails, and choose the sequence whose tally span is closer to their own expected span (with a personal left/right lean) — but they also directly register how often the coin switches sides, and a sequence that switches more often looks more random in its own right, even when its tally span is the same. The one change is a shared reward on each sequence's share of H/T switches (taken from the alternation-ideal models), addressing the critique (surviving FDR, q = 0.016) that among pairs with near-equal tally spans people pick the more-switching sequence far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tally_span_ideal_plus_switch_reward.py`

### tally_span_personal_ideal — pruned (experiment2 end of experiment)

**Margin:** 148.6 nats behind tally_span_switch_ideal_periodic_unit (5.0× dse)

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails, and judge randomness by how widely that tally swings over the whole reading — the span between the furthest it ever ran toward heads and the furthest it ever ran toward tails. Each person expects a fair coin's tally to wander over some typical span for a sequence of that length: a long streak (anywhere in the sequence, even one later evened out) stretches the span too far and strict alternation pins it within one step of even, so both look non-random, and people choose the sequence whose tally span is closer to their own expectation. Apart from this judgement, each person has a small habitual lean toward clicking left or right.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tally_span_personal_ideal.py`

### bayes_pattern_vs_alternating_coin — pruned (experiment1 end of experiment)

**Margin:** 153.5 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent it is better explained by a random coin than by a noisy repeating pattern (a short template of period 1-4, such as HHHH, HTHT or HHTT, copied with occasional errors). The single distortion is in their picture of the random coin: each person believes a fair coin switches sides at their own rate (often more than half the time), so alternation counts as evidence for randomness up to the point where the sequence becomes regular enough to be better explained by a repeating template.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/bayes_pattern_vs_alternating_coin.py`

### tally_excursion_triplet_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 156.3 nats behind tally_span_switch_ideal_periodic_unit (5.8× dse)

**Hypothesis:** Refinement of `iter1_candidate4` (switching rate close to a personal ideal, with person-specific length-scaled sensitivity, a periodicity penalty, a heads-default lean, a balance penalty and a reward for triplet variety): people still judge a sequence mainly by its switching rate, but while reading they also follow a running tally of how far heads lead tails, and a sequence whose tally wanders far from even at some point — a streak, even one later evened out — looks less random. The one change is a shared penalty on the span of that running tally (as a share of the length), grafted from the incumbent `tally_span_personal_ideal`; unlike the incumbent, switching is judged in its own right, addressing the critique (q = 0.016) that among pairs with nearly equal tally spans people pick the more-switching sequence far more often than a span-only model predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tally_excursion_triplet_alternation_ideal.py`

### iter1_candidate4 — pruned (experiment2 end of experiment)

**Margin:** 156.4 nats behind tally_span_switch_ideal_periodic_unit (5.7× dse)

**Hypothesis:** Refinement of the incumbent `balance_aware_heads_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences), still favour heads-leaning sequences and still find lopsided H/T counts less random — but they also register the variety of short local patterns, so a sequence whose overlapping three-flip chunks are mostly different from one another (rather than the same few chunks recurring) looks more random, even when its overall switching rate is the same. The one change is a shared reward on the share of distinct three-flip chunks a sequence contains, addressing the critique (surviving FDR, q = 0.008) that among pairs with equal switch counts people pick the sequence with more distinct triplets far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/iter1_candidate4.py`

### streak_aware_balanced_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 157.3 nats behind tally_span_switch_ideal_periodic_unit (5.9× dse)

**Hypothesis:** Refinement of the incumbent `balance_aware_heads_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences), still favour heads-leaning sequences and still find lopsided H/T counts non-random — but they also notice a streak of identical outcomes as a sign of non-randomness in its own right, so every flip of the longest run beyond two (HHH, HHHH, ...) makes a sequence look less random even when its overall switching rate is the same. The one change is a shared penalty on this absolute excess length of the longest streak, addressing the critique (surviving FDR, q = 0.008) that among pairs with equal switch counts people pick the sequence with the shorter longest run far more often than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/streak_aware_balanced_alternation_ideal.py`

### balance_aware_heads_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 162.1 nats behind tally_span_switch_ideal_periodic_unit (5.6× dse)

**Hypothesis:** Refinement of the incumbent `heads_default_alternation_ideal`: people still judge a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences) and still favour heads-leaning sequences, but they also expect a random coin to come out roughly half heads and half tails, so a sequence whose heads and tails counts are lopsided in either direction looks less random. The one change is a shared symmetric penalty on the H/T imbalance (|#H − #T| as a share of the length), addressing the critique that among pairs with equal switch counts people choose the more balanced sequence far more often than the incumbent's one-directional heads-share term allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/balance_aware_heads_alternation_ideal.py`

### balanced_heads_default_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 163.1 nats behind tally_span_switch_ideal_periodic_unit (5.8× dse)

**Hypothesis:** Refinement of the incumbent `heads_default_alternation_ideal`: each person still judges a sequence by how close its switching rate is to their own ideal (person-specific, length-scaled sensitivity, shared penalty for periodic sequences) and still treats heads as the default face, so heads-leaning sequences look more random — but people also expect a fair coin to produce roughly equal numbers of heads and tails, so a lopsided sequence in either direction looks less random. The one change is a shared symmetric penalty on the squared departure of the heads share from one half, addressing the critique (surviving FDR) that among pairs with equal switch counts people pick the more balanced sequence far more often than a model with only a linear heads-share term allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/balanced_heads_default_alternation_ideal.py`

### iter0_candidate0 — pruned (experiment2 end of experiment)

**Margin:** 164.8 nats behind tally_span_switch_ideal_periodic_unit (6.5× dse)

**Hypothesis:** People judge randomness by Bayesian model comparison: a sequence looks random to the extent that it is better explained by a random coin than by a "rigged" process — either a coin biased toward one face (of unknown bias) or a sticky/switchy coin that tends to repeat or alternate (of unknown tendency) — so lopsided H/T counts and extreme switching both count as evidence of a non-random generator. The one distortion is in each person's picture of the random coin itself: rather than a fair, memoryless coin, each person believes a random coin switches sides at their own personal rate (usually more than half the time), and they choose the sequence with the higher posterior odds of having come from that subjective random coin.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/iter0_candidate0.py`

### heads_default_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 169.2 nats behind tally_span_switch_ideal_periodic_unit (5.4× dse)

**Hypothesis:** People do not treat the two faces of the coin symmetrically: heads is the default, expected outcome of a coin toss, so a sequence dominated by tails reads as a coin that is "off" (biased toward the unusual face) and looks less random, while a heads-leaning sequence looks like ordinary coin flipping. This label asymmetry operates on top of each person's judgement of how close a sequence's switching rate is to their own ideal (with person-specific, length-scaled sensitivity and a shared penalty for visibly periodic sequences): of two sequences that switch equally often, people pick the one with more heads.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/heads_default_alternation_ideal.py`

### bayes_rigged_coin_with_two_flip_memory — pruned (experiment2 end of experiment)

**Margin:** 177.3 nats behind tally_span_switch_ideal_periodic_unit (5.5× dse)

**Hypothesis:** Refinement of `iter0_candidate0` (Bayesian detection of a rigged coin with a personally distorted random coin): people still judge a sequence as random by how much better a random coin (one that switches sides at their own personal rate) explains it than a rigged process does, and they still consider a coin biased toward one face and a sticky/switchy coin. The one change is a third rigged process they also entertain: a coin with a two-flip memory, whose next flip depends on the previous two (an unknown rule per two-flip context, such as "after HH comes H"). Sequences that reuse the same few triplets — long streaks, and short repeating patterns — are well explained by that memory coin and so look rigged even when their switch count matches the other sequence's, addressing the critique that at equal switch counts people pick the sequence with the shorter longest run and more distinct triplets.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/bayes_rigged_coin_with_two_flip_memory.py`

### balanced_heads_lapse_ideal_streak — pruned (experiment2 end of experiment)

**Margin:** 192.4 nats behind tally_span_switch_ideal_periodic_unit (6.4× dse)

**Hypothesis:** Refinement of `heads_favoring_lapse_ideal_streak`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own ideal switching rate, still penalises every flip of the longest streak beyond two, still favours heads-leaning sequences, and still lapses to a random pick at their own rate — but people also expect a fair coin to produce roughly equal numbers of heads and tails, so a lopsided H/T count (in either direction) looks less random. The one change is a shared penalty on the absolute heads–tails imbalance of each sequence, addressing the critique's strongest discrepancy: among pairs with equal switch counts, people choose the more balanced sequence far more often than a model with only a directional heads-share term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/balanced_heads_lapse_ideal_streak.py`

### heads_favoring_lapse_ideal_streak — pruned (experiment2 end of experiment)

**Margin:** 197.6 nats behind tally_span_switch_ideal_periodic_unit (6.8× dse)

**Hypothesis:** Refinement of `personal_lapse_ideal_streak_excess`: each person still judges a sequence as random by how close its proportion of H/T switches is to their own personal ideal switching rate, still penalises every flip of the longest streak beyond two, and still lapses to a random pick at their own rate — but the two coin faces are not psychologically symmetric: heads is the canonical, expected outcome of a coin toss, so a sequence with a larger share of heads reads as more like "real" coin flipping and looks more random. The one change is a shared heads-share bias inside the engaged decision, addressing the critique that among pairs with equal switch counts people choose the heads-heavier sequence more often than an H/T-symmetric model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/heads_favoring_lapse_ideal_streak.py`

### personal_markov_coin_belief — pruned (experiment1 end of experiment)

**Margin:** 205.7 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** Each person carries their own subjective model of what a fair coin does: a belief about how often a random coin switches between H and T from one flip to the next (many believe it switches more than half the time, some less). They judge which sequence is more random by how probable each sequence would be under their own believed coin, choosing between the two in proportion to those probabilities, so people differ systematically in how strongly — and in which direction — alternation makes a sequence look random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_markov_coin_belief.py`

### switching_then_balance_semiorder — pruned (experiment2 end of experiment)

**Margin:** 212.4 nats behind tally_span_switch_ideal_periodic_unit (7.9× dse)

**Hypothesis:** People compare the two sequences lexicographically (a semiorder): they first ask which one switches between H and T at a rate closer to their own personal ideal for a random coin; only when the two sequences switch about equally often does a second comparison take over, and then they pick the one whose heads and tails are more evenly balanced. Because the balance cue is consulted only when the partner leaves the switching comparison undecided, the same sequence's balance decides the choice beside a partner with the same number of switches and is ignored beside a partner that switches clearly more or less.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/switching_then_balance_semiorder.py`

### running_tally_drift_ideal — pruned (experiment1 end of experiment)

**Margin:** 234.8 nats behind graded_periodicity_personal_ideal (5.9× dse)

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails (or tails lead heads), and judge randomness by how far that tally wanders from balance along the way: each person expects a fair coin's tally to drift away from even by some typical amount (scaled to how many flips have been read so far), and a sequence looks random to the extent that its average drift matches that personal expectation. A long streak makes the tally run away from balance (too much drift), while strict alternation pins it at even (too little drift), so the path of the tally — not the final count or the switch rate — decides the choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/running_tally_drift_ideal.py`

### chunk_entropy_variety — pruned (experiment1 end of experiment)

**Margin:** 246.4 nats behind graded_periodicity_personal_ideal (4.1× dse)

**Hypothesis:** People judge randomness by the variety of short patterns a sequence contains: reading it as overlapping chunks of one to four flips, a sequence looks random to the extent that its chunks of each size are spread evenly over the different possible chunks (high chunk entropy for its length), and repetitive to the extent that the same few chunks recur — through a lopsided H/T count, a long streak, strict alternation, or an almost-repeating unit with one slip. People differ in how strongly this felt variety drives their choice. This disagrees most sharply with the switch-rate-ideal incumbent on pairs with similar switch counts where one sequence reuses its chunks (near-periodic sequences, or one with a long streak), and on block-structured sequences like HHTTHHTT, whose middling switch rate the incumbent rates as plausible but whose chunk variety is low.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/chunk_entropy_variety.py`

### personal_side_bias_alternation_ideal — pruned (experiment2 end of experiment)

**Margin:** 259.0 nats behind tally_span_switch_ideal_periodic_unit (8.7× dse)

**Hypothesis:** The decision rule, not the evidence, carries a stable individual trait: each person has their own habitual lean toward clicking the left or the right button, which adds to their impression of which sequence is more random on every trial (people differ in the direction and strength of this response-side bias, and it is unrelated to the sequences shown). The impression itself is kept simple: a sequence looks random to the extent that its proportion of H/T switches is close to the person's own ideal switching rate, so the side bias decides near-ties and shifts close calls toward the favoured side.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/personal_side_bias_alternation_ideal.py`

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

### max_lopsided_window_surprise — pruned (experiment2 end of experiment)

**Margin:** 413.4 nats behind tally_span_switch_ideal_periodic_unit (7.0× dse)

**Hypothesis:** People judge a sequence by its single most lopsided stretch: they scan every contiguous stretch of flips and register the one whose heads/tails split is most improbable for a fair coin of that stretch's length (a streak, or a stretch like HHTHHH that is nearly all one face), and that one most surprising stretch alone decides how non-random the sequence looks — the rest of the sequence does not count. They pick the sequence whose most lopsided stretch is less surprising, with people differing in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/max_lopsided_window_surprise.py`

### random_vs_designed_exemplar_contrast — pruned (experiment2 end of experiment)

**Margin:** 447.9 nats behind tally_span_switch_ideal_periodic_unit (10.2× dse)

**Hypothesis:** People judge randomness by exemplar memory of both kinds of sequence: they hold remembered examples of random-looking coin sequences (balanced heads and tails, no streak longer than two, no visible repeating unit) and of designed sequences (strings built by repeating a short unit at least twice, such as HHHH, HTHT, HHTHHT, HHTTHHTT), and a new sequence looks random to the extent that it is more similar to the random exemplars than to the designed ones, similarity falling off exponentially with the number of flips that differ. People pick the sequence with the higher random-versus-designed similarity, differing only in how strongly that felt resemblance drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/random_vs_designed_exemplar_contrast.py`

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

### run_length_profile_typicality — pruned (experiment2 end of experiment)

**Margin:** 640.6 nats behind tally_span_switch_ideal_periodic_unit (7.9× dse)

**Hypothesis:** People judge randomness by the mix of streak lengths a sequence breaks into: each person expects a random coin to produce a characteristic spread of runs — mostly single flips, fewer pairs, rarer triples and only occasionally longer streaks — and a sequence looks random to the extent that the proportions of its runs of length 1, 2, 3 and 4+ match that personal expected mix. Strict alternation (all single runs), uniform blocks like HHTTHHTT (all pairs) and long streaks all have run-length profiles far from the expected mix, and people differ both in the mix they expect and in how strongly a mismatch drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/run_length_profile_typicality.py`

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Margin:** 657.0 nats behind graded_periodicity_personal_ideal (9.7× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/finite_experience_occurrence.py`

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Margin:** 668.3 nats behind graded_periodicity_personal_ideal (10.6× dse)

**Hypothesis:** People judge randomness by exemplar memory: they hold remembered examples of "designed" coin sequences — streaks (HHHH), strict alternation (HTHT), and short repeating units (HHT, HHTT, HHHT, HTTT and their shifts and mirror images) — and a sequence looks non-random to the extent that it is similar to these stored patterns, with similarity falling off exponentially with the number of flips that would have to change to turn the sequence into a stored pattern (summed over all stored examples, as in a generalized context model). Random examples are remembered too diffusely to favour any particular sequence, so people pick the sequence with less summed similarity to the designed exemplars; individuals differ in how strongly the alternation exemplars are stored, so for some people near-alternating sequences look designed and for others they do not.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment1/model_loop/models/pruned/exemplar_designed_pattern_similarity.py`

### gist_typicality_count_runs — pruned (experiment2 end of experiment)

**Margin:** 734.1 nats behind tally_span_switch_ideal_periodic_unit (9.8× dse)

**Hypothesis:** People judge how random a sequence looks by how typical its gist is for a fair coin: they register only two coarse summaries of a sequence — how many heads it has and how many runs (streaks) it breaks into — and a sequence looks random to the extent that many coin sequences of that length share that same gist. Lopsided counts, long streaks (few runs) and strict alternation (maximal runs) are all rare gists and look non-random, while balanced sequences with a middling number of runs are common gists and look random; people differ only in how strongly this felt typicality drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/gist_typicality_count_runs.py`

### lempel_ziv_copy_complexity — pruned (experiment2 end of experiment)

**Margin:** 738.7 nats behind tally_span_switch_ideal_periodic_unit (9.8× dse)

**Hypothesis:** People judge randomness by how hard a sequence is to describe by copying: reading it from left to right, they mentally chunk it into the fewest pieces such that each new piece is either a single new flip or a copy of something already seen earlier in the sequence (Lempel–Ziv parsing), and a sequence that breaks into more such "new" pieces looks more random. Streaks and strict alternation (HTHTHTHT collapses to H, T and one long copy) and repeated units are cheap to describe and look non-random, while sequences full of varied local patterns resist copying and look random; people differ in how strongly this felt incompressibility drives their choice, and each has a small habitual lean toward clicking left or right.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/lempel_ziv_copy_complexity.py`

### max_copyable_motif_stretch — pruned (experiment2 end of experiment)

**Margin:** 771.0 nats behind tally_span_switch_ideal_periodic_unit (8.3× dse)

**Hypothesis:** People judge a sequence by its single most striking copyable stretch: they scan it for the longest contiguous stretch that keeps repeating one short unit of one to four flips (a streak like HHHH, strict alternation like HTHTHT, or a repeating motif like HHTHHT or HTTTHTTT), and register how many flips of that stretch could be predicted just by copying the unit — the one most predictable stretch alone decides how patterned the sequence looks, the rest of the sequence does not count. They pick the sequence whose most copyable stretch is shorter, with people differing in how strongly this felt pattern drives their choice and each having a small habitual left/right lean.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/max_copyable_motif_stretch.py`

### switch_after_repeat_markov_coin — pruned (experiment2 end of experiment)

**Margin:** 781.3 nats behind tally_span_switch_ideal_periodic_unit (12.7× dse)

**Hypothesis:** People judge randomness with a subjective coin that remembers only its last two flips: they expect a random coin's next flip to switch sides with a probability that depends on whether the previous two flips were a repeat or a switch (typically, a repeat should be followed by a switch, while a switch need not be followed by another switch), and a sequence looks random to the extent that it is probable under this two-flip-memory coin. People choose between the two sequences in direct proportion to how probable this subjective coin makes each one (a Luce choice on the probabilities themselves, with no separate decisiveness parameter), so long streaks (repeat after repeat) and strict alternation (switch after switch) both look non-random, apart from a small personal lean toward clicking left or right.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/switch_after_repeat_markov_coin.py`

### tie_return_rate_personal_ideal — pruned (experiment2 end of experiment)

**Margin:** 805.2 nats behind tally_span_switch_ideal_periodic_unit (10.8× dse)

**Hypothesis:** People read a sequence flip by flip while tracking whether heads and tails are currently even, and judge randomness by how often the coin "evens itself out" — how many times the running count of heads and tails comes back to a tie, as a share of the most ties the sequence could reach. Each person expects a fair coin to restore balance at their own typical rate: a streak or a lopsided sequence that rarely gets back to even looks rigged, and one that snaps back to even every two flips (HTHT..., HTTHHTTH) looks too self-correcting, so people choose the sequence whose rate of returning to a tie is closer to their own expectation. This disagrees most sharply with the tally-span model on pairs with equal spans but different return counts (e.g. HTTHHTTH, which ties four times, versus HHTTHHTT, which ties twice, or HHTHTTHT versus HTHHTHTT), and each person also has a small habitual lean toward clicking left or right.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/tie_return_rate_personal_ideal.py`

### bayes_triplet_chunk_rig_detector — pruned (experiment2 end of experiment)

**Margin:** 852.5 nats behind tally_span_switch_ideal_periodic_unit (10.2× dse)

**Hypothesis:** People judge randomness like an ideal Bayesian detector working over three-flip chunks: they ask whether a sequence's overlapping three-flip chunks look like even draws from all eight possible chunks (as a fair coin gives) or like draws from a rigged process that favours some chunks over others, and a sequence looks random to the extent that the fair-coin account explains its chunks better. The one distortion is a resource limit in how they encode the sequence: they only register it as a bag of three-flip chunks (ignoring how the chunks overlap and where they occur), so strict alternation (only HTH and THT), streaks (repeated HHH) and short repeating units all look rigged, while sequences with varied chunks look random; each person also has a small habitual left/right lean.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/pruned/bayes_triplet_chunk_rig_detector.py`
