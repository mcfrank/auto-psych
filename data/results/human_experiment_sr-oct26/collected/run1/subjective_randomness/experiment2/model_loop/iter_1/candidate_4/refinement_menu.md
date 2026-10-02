# Refinement menu

The models you may refine, other than the incumbent `terminal_run_streak_aversion_lapse`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### terminal_discounted_streak_lapse — rank 1, 2.7 ± 7.7 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -2381.7

**Hypothesis:** Refinement of the incumbent `individual_streak_aversion_lapse`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts and the longest streak by weights of their own, and still guesses on some trials at a personal lapse rate. The one change is that the streak a sequence ends on is weighed differently from streaks that have already been closed off: because sequences are read left to right, a run still in progress at the last flip is judged as an unfinished streak, so its length counts by a fitted shared factor (discounted or amplified) when deciding which streak is the sequence's most striking one — addressing the critique that people choose the sequence with the longer final run more often than the position-blind longest-run term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/terminal_discounted_streak_lapse.py`

### individual_streak_aversion_lapse — rank 2, 11.8 ± 9.4 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -2390.8

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that people also notice the longest streak of identical flips and treat it as a sign of non-randomness by a weight that is each person's own (drawn from a population distribution, so some people are strongly streak-averse and others barely care), addressing the critique that individuals differ in preferring the sequence with the shorter longest run more than the incumbent produces, and that streaks are penalised beyond its alternation and balance terms.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/individual_streak_aversion_lapse.py`

### edge_streak_balance_alternation_lapse — rank 3, 31.0 ± 28.7 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -2409.9

**Hypothesis:** Refinement of the incumbent `lapse_individual_balance_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate. The one change is that a streak sitting at the very start or end of a sequence is especially salient (primacy/recency of what is read first and last): the number of flips by which a run of identical flips at either edge exceeds two makes the sequence look less random by a shared weight, while the same streak buried in the middle carries no extra penalty — addressing the critique that among alternation-matched pairs people avoid the sequence with an edge streak more than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/edge_streak_balance_alternation_lapse.py`

### primacy_recency_streak_balance_lapse — rank 4, 60.8 ± 21.8 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2439.7

**Hypothesis:** Refinement of `edge_streak_balance_alternation_lapse` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own ideal switching rate, still penalises lopsided heads/tails counts by a weight of their own, and still guesses on some trials at a personal lapse rate, and streaks at the edges of a sequence still carry an extra penalty. The one change is that the opening streak and the closing streak are no longer treated as one interchangeable "edge" streak: the streak a person reads last (recency) and the streak they read first (primacy) each make the sequence look less random by a weight of its own, so a run at the end can count for more than the same run at the start — addressing the critique that people penalise a long final run more than the incumbent's position-blind longest-run term predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/primacy_recency_streak_balance_lapse.py`

### lapse_individual_streak_aversion — rank 5, 72.4 ± 18.5 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2451.3

**Hypothesis:** Refinement of `personal_lapse_ideal_alternation`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and still guesses on some trials at a personal, trait-like lapse rate, but people also treat a long streak of identical flips as a sign of non-randomness, and how strongly a streak puts them off is each person's own trait (a personal weight on the longest run relative to sequence length, drawn from a population). The one change is this individual streak-aversion weight, added because the critique shows participants differ in their preference for the sequence with the shorter longest run more than the model's alternation, balance and lapse heterogeneity produces, and that streaks are penalised beyond alternation among alternation-matched pairs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/lapse_individual_streak_aversion.py`

### lapse_individual_balance_alternation — rank 6, 88.1 ± 19.4 nats behind the best (4.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2467.1

**Hypothesis:** Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/lapse_individual_balance_alternation.py`

### shared_stretch_cancellation — rank 7, 154.0 ± 36.0 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2532.9

**Hypothesis:** People compare the two sequences side by side and cancel what they share: flips that both sequences have in common at their start and at their end are discounted as uninformative, and each sequence's randomness is judged only on the stretch where the two differ (plus the flip on either side of it, so its switches into and out of that stretch count). On that differing stretch each person still judges randomness by how close its switching rate lies to their own ideal, how lopsided its heads/tails mix is and how long its longest streak is, and guesses on some trials; so the same sequence is judged differently beside a partner that shares its opening or its ending than beside one that shares nothing.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/shared_stretch_cancellation.py`

### edge_weighted_alternation_memory — rank 8, 227.4 ± 34.5 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2606.3; PSIS-LOO unreliable (2% of trials with a high Pareto k) — its ELPD is untrustworthy

**Hypothesis:** People encode a coin-flip sequence with serial-position effects in memory: the flips at the start and end of the sequence (primacy and recency) are encoded more strongly than those in the middle, so the switching pattern they perceive is dominated by what happens at the sequence's edges. Each person compares this edge-weighted impression of how often the sequence switches between heads and tails with their own personal ideal switching rate, and on some trials guesses at a personal lapse rate; a streak or rigid alternation sitting at an edge therefore makes a sequence look much less random than the same pattern buried in the middle.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/edge_weighted_alternation_memory.py`

### lempel_ziv_incompressibility — rank 9, 514.4 ± 78.4 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2893.4

**Hypothesis:** People judge a coin-flip sequence's randomness by how hard it is to describe compactly: reading it left to right, they notice whenever the next stretch of flips merely copies something already seen, and a sequence looks random in proportion to how many genuinely new chunks it takes to spell it out (its Lempel-Ziv compression complexity). How strongly this incompressibility drives the choice is each person's own trait, drawn from a population, so streaks, rigid alternation and repeated motifs all count against a sequence only through the single fact that they make it compressible.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/lempel_ziv_incompressibility.py`

### leaky_memory_bayesian_randomness — rank 10, 825.2 ± 93.5 nats behind the best (8.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3204.1

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent a fair coin explains it better than the non-random generators one might suspect (a biased coin, or a sticky/switchy coin whose switching probability is unknown), with the marginal likelihood of each alternative computed from the sequence's head/tail and repeat/switch counts. The one distortion is leaky memory: as they read left to right, earlier flips fade, so the evidence each flip contributes is discounted geometrically with its distance from the end of the sequence, which makes a streak or rigid pattern at the end of a sequence count as much stronger evidence of non-randomness than the same pattern at its start (people also guess on some trials at a personal rate).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment2/model_loop/models/leaky_memory_bayesian_randomness.py`

## Pruned models (out of the set; narrowest margin first)

### individual_balance_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 58.0 nats behind individual_streak_aversion_lapse (4.5× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still make a sequence look less random, but people differ in how much that imbalance matters to them — each person has their own weight on H/T balance, drawn from a population distribution, instead of everyone sharing one. The one change is making the balance penalty individual, because the critique shows participants differ in their preference for the more balanced sequence far more than a single shared balance weight allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/individual_balance_ideal_alternation.py`

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

### length_scaled_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 78.2 nats behind individual_streak_aversion_lapse (5.2× dse)

**Hypothesis:** Refinement of the incumbent `ideal_alternation_with_balance`: each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts still look less random, but people treat a departure from their ideal switching rate as evidence that grows with the number of flips it is observed over — the same off-ideal alternation rate is shrugged off in a short sequence and taken seriously in a long one. The one change is making alternation sensitivity grow with sequence length (as a power of the number of transitions, fitted), because the critique shows short sequences are judged by alternation less, relative to long ones, than the length-independent incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/length_scaled_ideal_alternation.py`

### iter1_candidate1 — pruned (experiment1 end of experiment)

**Margin:** 82.0 nats behind individual_streak_aversion_lapse (4.8× dse)

**Hypothesis:** People judge randomness normatively, as Bayesian model comparison: a sequence looks random to the extent it is better explained by a fair coin than by the non-random generators one might suspect — a biased coin (which explains unbalanced heads/tails counts) or a sticky/switchy coin (which explains long runs or rigid alternation). The one distortion is that each person's mental model of a "fair" coin switches between heads and tails at their own personal rate rather than exactly half the time, so people disagree about alternation while all still penalise imbalance and runs that the alternative generators explain.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/iter1_candidate1.py`

### divisive_contrast_ideal_alternation — pruned (experiment1 end of experiment)

**Margin:** 84.3 nats behind individual_streak_aversion_lapse (5.3× dse)

**Hypothesis:** People do not judge each sequence's randomness on an absolute scale; they judge the two sequences against each other by divisive contrast: how much less random one looks than the other is weighed relative to how non-random the pair looks overall. Each sequence's non-randomness is its departure from the person's own ideal switching rate plus its heads/tails imbalance, but a given difference decides the choice strongly when the partner is nearly ideal and only weakly when both sequences are clearly non-random, so the same sequence is judged differently beside a different partner.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/divisive_contrast_ideal_alternation.py`

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

### personal_ideal_longest_run — pruned (experiment1 end of experiment)

**Margin:** 133.1 nats behind individual_streak_aversion_lapse (5.0× dse)

**Hypothesis:** People judge randomness by the longest streak a sequence contains: each person carries their own ideal length for the longest run of identical flips in a random sequence (relative to its length), and picks the sequence whose longest streak is closer to that personal ideal. Some people expect random sequences to contain almost no streaks, others expect a noticeable streak, so the same pair can be judged in opposite directions; the rest of the sequence's structure (its overall alternation rate or H/T balance) plays no role beyond what it does to the longest streak.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_ideal_longest_run.py`

### individual_irregularity_weight — pruned (experiment1 end of experiment)

**Margin:** 148.5 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** Refinement of the incumbent `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random when it is locally balanced and irregular — close to an over-alternating prototype and not periodic). The one change: people differ in *which* part of representativeness they rely on — each person has their own weight on irregularity (alternation and non-periodicity) versus local H/T balance, drawn from a population distribution — while the prototype alternation rate stays shared. Alternation-focused people then decide most pairs by how much the sequences switch and balance-focused people by their mix of heads and tails, which addresses the critique that participants differ far more in their preference for the more-alternating sequence than a single-population model produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/individual_irregularity_weight.py`

### personal_switch_belief — pruned (experiment1 end of experiment)

**Margin:** 159.2 nats behind individual_streak_aversion_lapse (6.0× dse)

**Hypothesis:** Each person carries their own subjective model of a fair coin as a process that switches between heads and tails with a personal probability (some believe coins alternate far more than half the time, others near or below half), and judges as more random whichever sequence is more probable under that personal switching belief. Because the believed switch rate differs from person to person, the same pair can be judged in opposite directions by different participants: strong alternation-lovers pick the more alternating sequence, others pick the streakier one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/personal_switch_belief.py`

### tally_drift_ideal — pruned (experiment1 end of experiment)

**Margin:** 186.0 nats behind individual_streak_aversion_lapse (5.9× dse)

**Hypothesis:** People keep a running tally of heads minus tails as they read a sequence, and judge it random to the extent that this tally wanders away from balance by about as much as they expect a fair coin's tally to wander: the whole path matters, so a streak that pushes the tally far off balance (even one later corrected) counts against a sequence, and so does a tally that never leaves balance at all. Each person has their own ideal amount of wandering — some expect the tally to hug zero, others tolerate large drifts — and picks the sequence whose tally drift is closer to that personal ideal.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/tally_drift_ideal.py`

### pattern_coverage_detection — pruned (experiment1 end of experiment)

**Margin:** 194.8 nats behind individual_streak_aversion_lapse (6.2× dse)

**Hypothesis:** People judge randomness by pattern detection: they scan a sequence for perceptible regularities — a streak of identical flips or a stretch of strict H/T alternation — and a sequence looks non-random in proportion to how much of it is covered by such a detected pattern. A stretch is noticed as a pattern once it is long enough, with streaks noticed at a shared length and alternating stretches at a length that differs from person to person, so the whole shape of the sequence (how many of its flips sit inside long runs or long alternations, not its overall switch rate) drives the choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/pattern_coverage_detection.py`

### exemplar_designed_pattern_similarity — pruned (experiment1 end of experiment)

**Margin:** 287.9 nats behind individual_streak_aversion_lapse (10.4× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously designed coin sequences — a solid streak (HHHH…), strict alternation (HTHT…), pairs (HHTT…), triples (HHHTTT…) and two halves (HHHH TTTT) — and a sequence looks non-random to the extent that it closely resembles (differs in few flips from) any of these remembered patterns, with similarity falling off exponentially with the share of mismatched flips. People differ in whether their remembered set of "designed" sequences prominently includes strict alternation, so some find alternating sequences contrived and others find them random-looking, while streak-like resemblance is penalised by everyone.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/exemplar_designed_pattern_similarity.py`

### local_representativeness — pruned (experiment1 end of experiment)

**Margin:** 467.0 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/local_representativeness.py`

### motif_stack — pruned (experiment1 end of experiment)

**Margin:** 485.2 nats behind individual_streak_aversion_lapse (11.9× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack.py`

### online_transition_surprise — pruned (experiment1 end of experiment)

**Margin:** 624.4 nats behind individual_streak_aversion_lapse (12.3× dse)

**Hypothesis:** People judge randomness by trying to predict each flip from the one before it as they read the sequence left to right, learning the transition tendencies (after H comes ... ; after T comes ...) on the fly from the flips seen so far; a sequence looks random to the extent it stays surprising to this online learner. Order matters: a sequence whose transitions become predictable early (a streak, a strict alternation, or any repeating transition habit) loses randomness even if its overall alternation rate is moderate, and how quickly the learner commits is governed by a single prior-strength parameter.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/online_transition_surprise.py`

### falk_konold_dp — pruned (experiment1 end of experiment)

**Margin:** 648.0 nats behind individual_streak_aversion_lapse (14.5× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/falk_konold_dp.py`

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Margin:** 816.9 nats behind individual_streak_aversion_lapse (12.5× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/pruned/finite_experience_occurrence.py`
