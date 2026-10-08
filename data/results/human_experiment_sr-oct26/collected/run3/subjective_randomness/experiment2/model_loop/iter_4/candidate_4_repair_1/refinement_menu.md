# Refinement menu

The models you may refine, other than the incumbent `second_order_motif_side_habit`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### face_bias_motif_side_habit — rank 1, 0.1 ± 3.4 nats behind the best (0.0× dse: statistically tied with the best), ELPD-LOO -2103.5

**Hypothesis:** Refinement of `chance_face_bias_motif_suspicion`: people still judge a sequence random to the extent their own model of a fair coin — with a person-specific believed switch rate and a shared belief that one face (fitted; the data suggest tails) comes up slightly more often — explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own readiness to suspect a motif), the evidence weighed per flip by a fitted power of length. The single change, grafted from `person_side_habit_switch_belief`, is that each person also has their own habitual leaning towards the Left or Right button, drawn from a population, which adds to the randomness evidence before the choice, so when the two sequences look about equally random a person's side habit decides — addressing the FDR-surviving critique that participants' proportions of Left choices vary more across people than the face-bias/second-order models produce, while keeping the tail-favouring chance coin that addresses the other FDR-surviving critique (people choose the head-heavier sequence less often than a symmetric model predicts).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/face_bias_motif_side_habit.py`

### face_biased_second_order_motif — rank 2, 0.1 ± 3.9 nats behind the best (0.0× dse: statistically tied with the best), ELPD-LOO -2103.6

**Hypothesis:** Refinement of the incumbent `second_order_chance_motif_suspicion`: people still judge a sequence random to the extent their own second-order model of a fair coin (a person-specific believed switch rate, shifted right after a switch by a shared fitted amount) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length. The single change, grafted from `chance_face_bias_motif_suspicion`, is that this imagined fair coin is also not label-symmetric: people believe it lands on one face (fitted; the data suggest tails) slightly more often, so among otherwise similar sequences the one leaning towards that face is better explained by chance and looks more random — addressing the FDR-surviving critique that people choose the head-heavier sequence less often than the H/T-symmetric incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/face_biased_second_order_motif.py`

### second_order_chance_motif_suspicion — rank 3, 1.5 ± 2.2 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2105.0

**Hypothesis:** Refinement of the incumbent `person_specific_motif_suspicion`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips, with each person's own suspicion of motifs), the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that people's model of a fair coin is second-order: right after a switch they expect a further switch to be less (or more) likely than usual, by a shared fitted amount, so a sequence that keeps switching flip after flip is improbable under "chance" beyond what its total switch count implies — addressing the critique that among already highly alternating sequences people prefer still more switches less than the incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/second_order_chance_motif_suspicion.py`

### chance_face_bias_motif_suspicion — rank 4, 1.6 ± 4.0 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -2105.0

**Hypothesis:** Refinement of the incumbent `person_specific_motif_suspicion`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips, with each person's own readiness to suspect a motif), the evidence weighed per flip by a fitted power of length. The single change is that people's model of a fair coin is not label-symmetric: they believe a fair coin lands on one face (fitted; the data suggest tails) slightly more often than the other, so among otherwise similar sequences the one leaning towards that face is better explained by chance and looks more random — addressing the critique (surviving FDR) that people choose the head-heavier sequence less often than the symmetric incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/chance_face_bias_motif_suspicion.py`

### occam_neglect_trick_coin_fit — rank 5, 1.6 ± 2.5 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -2105.0

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate, and a second-order belief about switching twice in a row) explains it better than the regular generators (a switch-biased coin, a biased coin, and a short motif repeated with slips, with each person's own suspicion of motifs). The single distortion is a neglect of Occam's razor: instead of averaging over every rate a trick coin might have, people partly credit the trick coins with whatever switch rate or heads rate best fits the sequence at hand (a fitted share between the normative average and the best fit), so any sequence whose rates are even mildly lopsided is readily "explained" by a tailor-made trick coin, and moderately imbalanced or moderately streaky sequences look less random than the normative account implies, relative to perfectly typical ones.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/occam_neglect_trick_coin_fit.py`

### position_weighted_switch_impression — rank 6, 2.1 ± 2.3 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -2105.5

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific believed switch rate, and a second-order belief about switching twice in a row) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with slips, with each person's own suspicion of motifs), weighing the evidence per flip by a fitted power of length. The single new claim is about the order of reading: people read the sequence left to right and their felt sense of how often it switches is not position-blind — switches and repeats near the end of the sequence (recency) or near its start (primacy, fitted direction and strength) weigh more in that impression, so two sequences with the same switch count but with their streak at different places look differently random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/position_weighted_switch_impression.py`

### person_specific_motif_suspicion — rank 7, 2.3 ± 2.8 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -2105.7

**Hypothesis:** Refinement of the incumbent `person_specific_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that people also differ in how readily they suspect a repeating pattern: each person gives the repeating-motif generator their own prior weight among the regular explanations, drawn from a population, so for pattern-suspicious people perfect alternation and period-3/4 motifs look strongly non-random while pattern-blind people judge such sequences mainly by their switches and balance.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_specific_motif_suspicion.py`

### heads_rigged_trick_coin_suspicion — rank 8, 2.8 ± 3.5 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -2106.2

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators they suspect (a switch-biased Markov coin, a trick coin, and a repeating short motif with slips, with each person's own readiness to suspect a motif), weighing the evidence per flip by a fitted power of length. The single claim is that the trick coin people imagine is not symmetric: they suspect a coin rigged towards heads (the side a cheat would favour) more than one rigged towards tails, so a surplus of heads is taken as evidence of rigging while an equal surplus of tails is not. The model disagrees most sharply with the current best (H/T-symmetric) model on pairs of mirror-image or count-mismatched sequences, e.g. HHTHHH versus TTHTTT, where it predicts the tail-heavy sequence is chosen as more random — addressing the FDR-surviving critique that people choose the sequence with more heads less often than a symmetric model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/heads_rigged_trick_coin_suspicion.py`

### person_side_habit_switch_belief — rank 9, 3.1 ± 4.7 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2106.5

**Hypothesis:** Refinement of the incumbent `person_specific_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out, the evidence weighed per flip by a fitted power of length, and person-specific decisiveness. The single change is that each person also has their own habitual leaning towards the left or the right response, drawn from a population, which adds to the randomness evidence before the choice — so when the two sequences look about equally random a person's side habit decides — addressing the critique that participants' proportions of Left choices vary more across people than the incumbent produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_side_habit_switch_belief.py`

### heads_rigged_coin_suspicion — rank 10, 3.2 ± 4.8 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -2106.6

**Hypothesis:** Refinement of `person_side_habit_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the evidence weighed per flip, person-specific decisiveness and a person-specific side habit. The single change is in what people imagine a "biased coin" to be: they do not suspect heads-rigged and tails-rigged coins equally, but hold a lopsided prior (of fitted direction and strength) that trick coins are rigged towards heads, so a head-heavy sequence is more readily explained as coming from a rigged coin and looks less random than the mirror-image tail-heavy sequence — addressing the critique (surviving FDR) that people choose the sequence with more heads less often than an H/T-symmetric model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/heads_rigged_coin_suspicion.py`

### attention_gradient_sequential_evidence — rank 11, 3.3 ± 2.7 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -2106.7

**Hypothesis:** People judge randomness by reading a sequence flip by flip and accumulating evidence for "fair coin" against the regular explanations they suspect (a switch-biased coin, a biased coin, a repeating short motif with slips — their own believed switch rate of chance and their own suspicion of motifs included), but their attention is not spread evenly over the sequence: the evidence each flip contributes is weighted by an attention gradient along the reading order whose direction and steepness are fitted (primacy if early flips dominate, recency if late flips do). So the same streak, imbalance or pattern break makes a sequence look more or less random depending on where in the sequence it sits, and the model disagrees with position-blind accounts most on pairs that contain the same flips in a different order (e.g. a streak at the start versus at the end).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/attention_gradient_sequential_evidence.py`

### person_specific_chance_switch_belief — rank 12, 3.9 ± 4.6 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -2107.3

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that each person holds their own belief about how often a fair coin switches, drawn from a population distribution, instead of everyone sharing one over-alternating belief — so some people believe chance switches a lot while others believe chance is streaky and choose the sequence with fewer switches, addressing the critique that far more participants prefer fewer-switch sequences than the incumbent's shared belief allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_specific_chance_switch_belief.py`

### switch_belief_habitual_side_lapse — rank 13, 6.5 ± 5.3 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -2110.0

**Hypothesis:** Refinement of `personal_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often chance switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the evidence weighed per flip and person-specific decisiveness. The single change is that on a fraction of trials people do not make the judgment at all and simply click their own habitual side (each person has their own default side), so each person's choices are pulled towards their preferred button by a fixed share even on clear-cut pairs — unlike an additive side bias, which matters only when the evidence is weak — addressing the critique that participants' individual rates of choosing Left vary more than a model without response habits produces.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/switch_belief_habitual_side_lapse.py`

### personal_chance_switch_belief — rank 14, 6.8 ± 6.3 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -2110.2

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their own model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip. The single change is that the one distortion of the normative account — the believed switch rate of a fair coin — is held differently by each person, drawn from a population: most people believe chance over-alternates, but some believe chance is streaky, so they credit sequences with fewer switches as more random, addressing the critique that a sizeable minority of participants systematically prefer the sequence with fewer switches.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/personal_chance_switch_belief.py`

### person_specific_chance_switch_belief_2 — rank 15, 23.6 ± 9.0 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2127.0

**Hypothesis:** Refinement of `bayesian_chance_vs_motif_beta_trick_coin`: people still judge a sequence random to the extent their own model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin with a fitted-concentration prior, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is that each person holds their own belief about how often a fair coin switches sides, drawn from a population distribution, instead of everyone sharing one over-alternating belief: most people expect chance to over-alternate, but some believe a fair coin tends to repeat itself, so for them streakier sequences look more random — addressing the critique (surviving FDR) that far more participants prefer the sequence with fewer switches than a shared-belief model can produce.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_specific_chance_switch_belief_2.py`

### person_side_bias_chance_vs_motif — rank 16, 213.6 ± 32.7 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2317.0

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their over-alternating model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out, the evidence weighed per flip, and person-specific decisiveness. The single change is that each person also has their own habitual leaning towards the left or the right response, drawn from a population, which adds to the randomness evidence before the choice — so pairs with weak evidence are decided largely by a person's side habit and the same evidence yields different choice rates across people, addressing the critique that participants' rates of choosing the left sequence vary more than the incumbent allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_side_bias_chance_vs_motif.py`

### length_normalised_chance_vs_motif — rank 17, 214.9 ± 32.3 nats behind the best (6.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2318.3

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out. The single change is in how that evidence is weighed: people judge the evidence per flip rather than in total, so the log evidence difference is divided by a fitted power of the sequence length — the same per-flip regularity is judged about as decisively in a short pair as in a long one, instead of long sequences automatically yielding near-certain choices — addressing the critique that people are more decisive on short pairs (relative to long) than the incumbent predicts and that it over-penalises long perfect alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/length_normalised_chance_vs_motif.py`

### posterior_verdict_contrast — rank 18, 218.0 ± 32.5 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2321.4

**Hypothesis:** People do not compare the two sequences' raw weight of evidence; each sequence is first turned into a felt verdict, "how likely is it that this one came from a fair coin rather than from some regular generator" (a fair coin believed to over-alternate, against a switch-biased coin, a biased coin and a repeating short motif), and the choice is driven by the difference between the two verdicts. Because verdicts saturate, the same difference in evidence is decisive when the two sequences are both ambiguous and barely matters when both clearly look random or both clearly look regular, so a sequence's pull on the choice depends on its partner.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/posterior_verdict_contrast.py`

### bayesian_chance_vs_motif_beta_trick_coin — rank 19, 223.7 ± 33.1 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2327.2

**Hypothesis:** Refinement of the incumbent `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with each generator's unknowns averaged out. The single change is in what people imagine a "biased coin" to be: instead of a uniform prior over its heads rate, they hold a symmetric prior of fitted concentration — imagining either only grossly lopsided trick coins or mildly biased ones — which sets how much a modest heads/tails imbalance counts as evidence against chance, addressing the critique that people weight heads/tails balance more than the incumbent's biased-coin generator implies among sequences with similar switch counts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/bayesian_chance_vs_motif_beta_trick_coin.py`

### bayesian_chance_vs_repeating_motif_2 — rank 20, 235.7 ± 33.4 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2339.1

**Hypothesis:** Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent a fair coin (which they believe over-alternates) explains it better than a "regular" generator, with the regular generators' unknowns averaged out normatively. The single change is one more regular generator in their hypothesis space: a "repeating motif" process that writes a short pattern (one to four flips long, e.g. H, HT, HHT, HHTT) and keeps copying it with an occasional slip (a fitted slip rate), so sequences that are near-repetitions of a short motif — perfect alternation and period-3/4 patterns above all — are explained as regular and look less random, addressing the critique that the incumbent under-penalises perfect alternation and periodic motifs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/bayesian_chance_vs_repeating_motif_2.py`

### mismatch_noise_repeating_motif — rank 21, 236.7 ± 33.5 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2340.1

**Hypothesis:** Refinement of `mismatch_noise_paired_comparison`: people still compare the two sequences flip against flip, so matching positions cancel and the noise in the comparison grows with the number of positions at which the two sequences differ (making short and near-identical pairs judged more decisively), and each sequence's evidence is still the Bayesian "fair coin believed to over-alternate versus a regular generator" score. The single change is to the regular generators people entertain: besides a switch-biased and a heads-biased coin, they also consider a repeating-motif process that copies a short pattern (one to four flips) with an occasional slip, so near-periodic sequences such as perfect alternation or HHT-HHT are recognised as regular and look less random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/mismatch_noise_repeating_motif.py`

### bayesian_chance_vs_repeating_motif — rank 22, 250.2 ± 33.5 nats behind the best (7.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2353.7

**Hypothesis:** Refinement of the incumbent `bayesian_overalternating_chance_model`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators, with every generator's unknowns averaged out normatively. The single change is one more regular generator in the comparison — a "repeating-pattern" generator that picks a short motif (one to four flips) and repeats it — given a fitted share of the prior on regularity; so exactly periodic sequences (perfect alternation HTHT…, but also HHTHHT…, HHTTHHTT…) are recognised as patterns and condemned, addressing the critique that the incumbent over-credits perfect alternation and under-penalises period-3/4 motifs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/bayesian_chance_vs_repeating_motif.py`

### hot_hand_only_suspicion — rank 23, 252.7 ± 36.2 nats behind the best (7.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2356.2

**Hypothesis:** People judge a sequence random to the extent a plain fair coin explains it better than the only kinds of "non-random" coin they ever suspect — a hot-hand (streaky) coin that tends to repeat its last outcome by an unknown amount, or a coin biased towards heads or tails by an unknown amount — with those unknowns averaged out and the evidence weighed per flip. Switching is never suspicious in itself, so perfect alternation and period-3/4 patterns count as maximally random: this is where the model disagrees most sharply with the current best model, which condemns them as repeating motifs, while among sequences with similar switch counts it still condemns streaks and heads/tails imbalance.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/hot_hand_only_suspicion.py`

### mismatch_noise_paired_comparison — rank 24, 303.4 ± 38.7 nats behind the best (7.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2406.9

**Hypothesis:** People do not evaluate each sequence on its own and then compare two independent impressions; they compare the two sequences flip against flip, so positions where the two show the same outcome cancel out and only the mismatching positions carry evidence — and also noise — into the judgment. Each sequence's randomness evidence is the Bayesian "fair coin (believed to over-alternate) versus regular generator" score, but the noise in comparing them grows with the number of positions at which the two sequences differ, so the same difference in randomness is judged decisively beside a near-identical partner (and on short pairs) and hesitantly beside a very different one.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/mismatch_noise_paired_comparison.py`

### personal_second_order_chance_typicality — rank 25, 466.4 ± 59.5 nats behind the best (7.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2569.8

**Hypothesis:** People judge randomness by typicality under their own imagined chance process alone, without weighing any rival "regular" explanation: each person carries a personal second-order picture of how a fair coin behaves — how likely it is to switch right after a repeat, and how likely right after a switch — and a sequence looks random to the extent that this imagined coin would readily produce it (its per-flip probability under that picture). Because the switch expectation depends on whether the previous step was a repeat or a switch, someone who believes chance rarely switches twice in a row condemns long perfect alternation, while someone who believes repeats quickly give way to switches condemns streaks, and people differ in both beliefs.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/personal_second_order_chance_typicality.py`

### gist_tail_probability_test — rank 26, 475.6 ± 71.2 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2579.0

**Hypothesis:** People judge randomness like an intuitive significance test on a sequence's gist rather than by comparing explanations: they summarise each sequence by its head count, its number of switches and its longest run, and ask how often a fair coin (which they believe switches somewhat more than half the time) would produce a gist at least as rare as this one. A sequence looks random to the extent that this tail probability is high — a typical gist is unremarkable, while a rare gist (a lopsided count, a long streak, or too-perfect alternation) "rejects chance" — and people differ in how decisively this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/gist_tail_probability_test.py`

### person_signed_switch_preference — rank 27, 701.1 ± 81.9 nats behind the best (8.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2804.5

**Hypothesis:** People judge randomness by a single gist cue — how often a sequence switches between heads and tails — but they disagree about which direction is random: most people take frequent switching as the signature of a fair coin, while some hold the opposite intuition that streaky, clumpy sequences are what real chance looks like. Each person therefore has their own signed preference for switching, drawn from a population that can straddle zero, and picks the sequence whose switch rate their own preference favours.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_signed_switch_preference.py`

### switch_indifference_side_default — rank 28, 704.1 ± 82.3 nats behind the best (8.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2807.5

**Hypothesis:** People commit to a "more random" choice only when the two sequences differ clearly in their single gist cue, how often they switch between heads and tails (each person with their own signed preference for switching); when the difference in switch rate falls inside an indifference band, they feel no real preference and fall back on their own habitual response side instead of judging. Side habits therefore show up mainly on pairs with similar switch counts, which makes people's overall rates of choosing Left vary more than a judgment-only model allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/switch_indifference_side_default.py`

### longest_patterned_stretch — rank 29, 744.1 ± 55.4 nats behind the best (13.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2847.5

**Hypothesis:** People judge randomness by the single most striking patterned stretch in a sequence: they scan for the longest contiguous stretch that keeps copying one short motif — a streak (H, H, H…), strict alternation (HT, HT…), or a repeated three- or four-flip motif — and how many flips that one stretch "predicts" sets how non-random the whole sequence looks, whatever the rest of it does. Each kind of motif has its own salience (a run of copies of a streak need not be as damning as an equally long alternation), and the sequence whose most striking patterned stretch is weaker is chosen as more random.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/longest_patterned_stretch.py`

### exemplar_random_vs_designed_ratio — rank 30, 901.6 ± 97.9 nats behind the best (9.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3005.0

**Hypothesis:** People judge randomness by exemplar categorisation: they carry remembered examples of both kinds of sequence — "designed" ones (streaks, perfect alternation, and short three- or four-flip motifs repeated) and "random-looking" ones (balanced heads and tails, no long streak, not perfectly alternating) — and a sequence looks random to the extent its summed similarity to the random exemplars outweighs its summed similarity to the designed exemplars, with similarity falling off exponentially with the number of mismatching flips so that the nearest exemplars dominate. The sequence with the higher random-versus-designed similarity ratio is chosen, with people differing only in how decisively they act on it.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/exemplar_random_vs_designed_ratio.py`

### tally_reverting_chance_coin — rank 31, 1154.5 ± 144.7 nats behind the best (8.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3257.9

**Hypothesis:** People read a sequence flip by flip while keeping a running tally of how far heads lead tails, and they believe a fair coin keeps that tally in check: whenever one side is ahead, the next flip is expected to favour the side that is behind, more strongly the bigger the lead (a gambler's-fallacy pull back to balance acting on the cumulative count, not on the last flip). A sequence looks random to the extent its tally path is probable under this self-correcting coin — so early drifts that are corrected look random, while a lead that keeps growing or is left uncorrected looks non-random even when the final count is balanced — and people differ in how strongly (or even in which direction) they expect the tally to be pulled back.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/tally_reverting_chance_coin.py`

### symmetry_generator_suspicion — no comparison row

**Hypothesis:** People judge a sequence random to the extent their own model of a fair coin explains it better than the regular ways a sequence could have been made — and among those regular ways they entertain a kind no current model considers: a sequence built by symmetry, whose second half is the first half mirrored (a palindrome, HHTTTTHH), mirrored with the faces swapped (HHTHTHTT), or repeated with the faces swapped (HTTHTHHT), copied with occasional slips. So sequences that are globally symmetric look designed and less random, even when their switch count, heads/tails balance and short periodic motifs are unremarkable (everything else as in the current best account: a person-specific believed switch rate and second-order switch belief, a switch-biased coin, a biased coin, a repeating short motif with each person's own motif suspicion, evidence weighed per flip, person-specific side habit).

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/symmetry_generator_suspicion.py`

### heads_rigged_second_order_side_habit — no comparison row

**Hypothesis:** Refinement of the incumbent `second_order_motif_side_habit`: people still judge a sequence random to the extent their own second-order model of a fair coin (a person-specific believed switch rate, shifted right after a switch by a shared amount) explains it better than the regular generators they suspect (a switch-biased Markov coin, a trick coin, and a repeating short motif with slips, with each person's own suspicion of motifs), the evidence weighed per flip by a fitted power of length and each person's side habit deciding near-ties. The single change, taken from `heads_rigged_coin_suspicion`, is that the trick coin people imagine is not symmetric: they suspect coins rigged towards heads more than coins rigged towards tails (fitted direction and strength), so a surplus of heads is more readily explained away as rigging than an equal surplus of tails, and head-heavy sequences look less random than their tail-heavy mirrors — most strongly for lopsided sequences — addressing the FDR-surviving critique that people choose the head-heavier sequence less often than the H/T-symmetric incumbent predicts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/heads_rigged_second_order_side_habit.py`

## Pruned models (out of the set; narrowest margin first)

### run_length_gamblers_chance_vs_motif — pruned (experiment1 end of experiment)

**Margin:** 0.0 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is in their model of chance itself: instead of a fixed belief that a fair coin switches more often than half the time, people hold a gambler's-fallacy belief that the coin becomes more likely to switch the longer the current run has lasted (a fitted increase per extra flip in the run), so a long streak is less probable under "chance" than the same number of switches spread as short runs, and runs of length one (alternation) are what chance is expected to keep producing — addressing the critique that the incumbent under-penalises long streaks among sequences with equal switch counts and over-penalises perfect alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/run_length_gamblers_chance_vs_motif.py`

### gamblers_chance_vs_motif — pruned (experiment1 end of experiment)

**Margin:** 0.9 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips, unknowns averaged out), with the evidence weighed per flip by a fitted power of sequence length and person-specific sensitivity. The single change is to their model of chance itself: instead of a fair coin that switches with one fixed (over-alternating) probability, people hold a gambler's-fallacy coin whose chance of switching grows with the length of the current run, so under "chance" a long streak is improbable beyond what its switch count implies, while strict alternation (every run of length one) earns only the baseline switch rate — addressing the critique that the incumbent under-penalises long streaks among sequences with equal switch counts and over-credits perfect alternation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/gamblers_chance_vs_motif.py`

### simplicity_weighted_motif_chance — pruned (experiment1 end of experiment)

**Margin:** 7.1 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of `bayesian_chance_vs_repeating_motif_2`: people still judge a sequence random to the extent their (over-alternating) model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif copied with occasional slips), with the generators' unknowns averaged out. The single change is in which repeating motifs people consider likely: instead of treating every motif period (1 to 4 flips) as equally plausible, they hold a simplicity prior whose fitted decay makes shorter periods — above all a single repeated flip, i.e. a streak — more expected regular patterns than alternation or longer motifs, so long streaks are condemned more and perfect alternation less, addressing the critique that the best model over-penalises perfect alternation and under-penalises long runs among sequences with similar switch counts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/simplicity_weighted_motif_chance.py`

### switch_prototype_person_lapse — pruned (experiment1 end of experiment)

**Margin:** 26.1 nats behind length_normalised_chance_vs_motif (4.3× dse)

**Hypothesis:** People all judge randomness the same way — by how close a sequence's rate of switching between heads and tails is to their internal ideal switch rate — and apply that judgment with one shared, sharp decision rule; what differs between people is a lapse rate: on some fraction of trials (person-specific) a participant does not use the judgment at all and picks a side at random. Disagreement with the majority therefore comes from occasional inattentive coin-flip choices, which cap how decisive anyone's choices can be (even for clear-cut pairs such as perfect alternation versus a long streak), rather than from graded differences in sensitivity.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/switch_prototype_person_lapse.py`

### bayesian_overalternating_chance_model — pruned (experiment1 end of experiment)

**Margin:** 29.4 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 24

**Hypothesis:** People judge randomness as Bayesian inference: a sequence looks random to the extent a fair coin explains it better than a "regular" generator (a coin with an unknown tendency to switch or repeat, or a coin with an unknown bias towards heads or tails), with the regular generators' unknown rates averaged out normatively. The one distortion is in their model of chance itself: people believe a fair coin switches sides more often than half the time, so their "random" likelihood rewards alternation and penalises repeats by a fitted amount — which also sets how much a perfectly alternating sequence is still credited as random rather than condemned as regular.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/bayesian_overalternating_chance_model.py`

### iter0_candidate3 — pruned (experiment1 end of experiment)

**Margin:** 30.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 24

**Hypothesis:** Refinement of `local_representativeness`: people judge randomness by the same Kahneman & Tversky local-representativeness score (multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates), but they differ in how decisively they apply it. The single change is that the decision sensitivity (beta) is person-specific, drawn from a population distribution, rather than shared by everyone — addressing the critique that people differ in agreement with the majority far more than one pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/iter0_candidate3.py`

### iter0_candidate4 — pruned (experiment1 end of experiment)

**Margin:** 31.2 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 24

**Hypothesis:** Refinement of `local_representativeness` (Kahneman & Tversky's local representativeness: a sequence looks random to the extent it is locally balanced at several window scales and irregular, i.e. somewhat over-alternating but not periodic). The one change: people share this representativeness criterion but differ in how consistently they apply it, so each participant has their own decision sensitivity (drawn from a population distribution) rather than one shared sensitivity — addressing the critique that participants differ in agreement with the majority far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/iter0_candidate4.py`

### asymmetric_alternation_representativeness — pruned (experiment1 end of experiment)

**Margin:** 31.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific sensitivity): people judge randomness by the same score — multiscale local balance plus irregularity relative to an over-alternating prototype and periodic templates — but the deviation from their prototype alternation rate is felt asymmetrically. The single change: too few alternations (streaky, repetitive sequences) look strongly non-random, whereas too many alternations (up to perfect HTHT alternation) are penalised only by a fitted fraction of that slope, because over-alternation is what people expect of chance; this addresses the critique that the incumbent over-penalises perfectly alternating sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/asymmetric_alternation_representativeness.py`

### global_local_balance_representativeness — pruned (experiment1 end of experiment)

**Margin:** 31.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same score — H/T balance plus irregularity relative to an over-alternating prototype and periodic templates — but the grain at which they check balance is fitted rather than fixed. The single change: instead of averaging whole-sequence balance and short-window (2–4 flip) balance with equal fixed weights, people give a fitted share of their balance judgment to the sequence's overall heads/tails balance and the rest to local window balance, addressing the critique that people weight global H/T balance more than current models imply among sequences with similar switch counts.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/global_local_balance_representativeness.py`

### iter1_candidate4 — pruned (experiment1 end of experiment)

**Margin:** 33.3 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** Refinement of the incumbent `iter0_candidate3` (Kahneman & Tversky local representativeness with person-specific decision sensitivity): people judge randomness by the same multiscale local balance plus irregularity score, but their sense of "periodic pattern" only picks up repeating templates longer than two flips (e.g. HHT-HHT, HHTT-HHTT); streaks (period 1) and plain alternation (period 2) are judged only through the balance and alternation-rate cues, not penalised a second time as periodic patterns. The single change is this non-redundant periodicity penalty, addressing the critique that the incumbent over-penalises perfect alternation (and, more weakly, long streaks) relative to what people choose.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/iter1_candidate4.py`

### adaptive_transition_learner_surprise — pruned (experiment1 end of experiment)

**Margin:** 33.9 nats behind length_normalised_chance_vs_motif (4.6× dse)

**Hypothesis:** People judge randomness by reading each sequence flip by flip while an online pattern learner tries to predict whether the next flip will repeat or switch, starting from a prior expectation (which may favour switching) and updating that expectation from the transitions seen so far. A sequence looks random to the extent this learner keeps being surprised — so sequences whose transitions become predictable as you read them (long streaks, but also perfect alternation once it has been seen a few times) look less random — and people differ in how decisively this felt unpredictability drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/adaptive_transition_learner_surprise.py`

### ideal_switch_rate_prototype — pruned (experiment1 end of experiment)

**Margin:** 37.8 nats behind length_normalised_chance_vs_motif (4.8× dse)

**Hypothesis:** People judge randomness by a single gist cue: how often the sequence switches between heads and tails. They hold an internal ideal switch rate (a fair coin's "should look like" rate, which may be above one half), and a sequence looks random to the extent its switch rate is close to that ideal — too few switches (long streaks) and too many switches (perfect alternation) both make it look less random. People differ in how decisively this one cue drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/ideal_switch_rate_prototype.py`

### goldilocks_gamblers_surprise_v2 — pruned (experiment1 end of experiment)

**Margin:** 38.5 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 24

**Hypothesis:** Refinement of `recency_weighted_gamblers_surprise`: people still read each sequence flip by flip with a gambler's-fallacy expectation (the longer the current run, the more they expect it to break), with recent flips weighing more, but a sequence looks random when its felt surprise is close to the moderate level they expect from a real coin, not when it is minimal. The single change is that randomness is a concave ("just-right") function of the recency-weighted surprise rather than its plain absence, so sequences that are too predictable under the gambler's expectation — above all perfect alternation, which confirms every predicted reversal — look contrived, addressing the critique that the incumbent picks perfectly alternating sequences more often than people do.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/goldilocks_gamblers_surprise_v2.py`

### recency_weighted_gamblers_surprise — pruned (experiment1 end of experiment)

**Margin:** 43.2 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by reading each sequence flip by flip while predicting the next flip with a gambler's-fallacy expectation — the longer the current run, the more they expect it to break — and a sequence looks random to the extent its flips were unsurprising under that expectation. Their memory of the surprise fades, so surprises near the end of the sequence (such as a final repeat that extends a run) weigh more than early ones; people differ in how strongly this felt surprise drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/recency_weighted_gamblers_surprise.py`

### gamblers_fallacy_leaky_predictor — pruned (experiment1 end of experiment)

**Margin:** 43.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by running a gambler's-fallacy predictor through the sequence: before each flip they expect the coin to "correct" the heads/tails imbalance seen so far, with recent flips weighing more than older ones in that leaky running tally, and a sequence looks random to the extent its flips match those corrective expectations (high average predictive probability). Unlike the incumbent's position-blind balance and alternation summaries, this makes the *order* of flips matter — a terminal repeat or a late streak (which violates a strong, just-built expectation of reversal) is penalised far more than the same repeat early on — so the two models disagree most on equal-composition pairs that differ only in where a repeat or run sits, especially at the end; people also differ in how sharply they apply this judgment.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/gamblers_fallacy_leaky_predictor.py`

### sequential_pattern_alarm_survival — pruned (experiment1 end of experiment)

**Margin:** 60.9 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness with a sequential pattern alarm: while reading a sequence flip by flip, at each flip they may notice "this is a pattern", with a detection probability that rises with how long the current predictable stretch has lasted — a run of identical flips, or a chain of strict alternation — and alternation chains must persist longer before they trigger the alarm than streaks do, because people expect a fair coin to switch often. A sequence looks random to the extent it can be read to the end without the alarm firing, so a single long streak condemns a sequence even when its overall switch count is ordinary, while long perfect alternation is condemned only weakly; people differ in how decisively this drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/sequential_pattern_alarm_survival.py`

### streak_tolerance_alarm — pruned (experiment1 end of experiment)

**Margin:** 77.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness with a streak alarm: there is a tolerance for how long a run of identical flips may be before it "looks too long to be chance", and every run in a sequence that exceeds that tolerance raises the alarm, more so the further it exceeds it, while runs within the tolerance cost nothing. The sequence raising the smaller alarm is chosen as more random, and people differ in how strongly the alarm drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/streak_tolerance_alarm.py`

### motif_stack_person_sensitivity — pruned (experiment1 end of experiment)

**Margin:** 95.7 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: ELPD-LOO rank 23 of 24

**Hypothesis:** Refinement of `motif_stack`: people judge a sequence random to the extent it is poorly explained by the Griffiths et al. (2018) four-motif stack automaton (mirror, complement and duplication production methods) relative to a fair coin, exactly as in `motif_stack` — but each person applies that shared regularity-detection process with their own decisiveness. The single change is that the sensitivity mapping the randomness-score difference to choice varies across participants (a hierarchical, log-normal population of per-person sensitivities) instead of being one shared value, addressing the critique that people differ in how consistently they agree with the majority judgment far more than a single pooled sensitivity allows.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack_person_sensitivity.py`

### most_lopsided_window_alarm — pruned (experiment1 end of experiment)

**Margin:** 107.8 nats behind run_length_gamblers_chance_vs_motif; retired to keep the live set at 8 models: its fit cannot be trusted (unreliable PSIS-LOO or no convergence)

**Hypothesis:** People judge randomness by scanning a sequence for its single most lopsided stretch: among all contiguous windows of every length, they find the one whose heads/tails imbalance would be most surprising for a fair coin, and that one striking stretch alone sets how non-random the sequence looks. The sequence whose worst stretch is less surprising is chosen as more random, so locally balanced sequences (including perfect alternation) look random while any single streak or heavily one-sided patch condemns a sequence regardless of how balanced the rest is; people differ in how decisively this drives their choice.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/most_lopsided_window_alarm.py`

### local_representativeness — pruned (experiment1 end of experiment)

**Margin:** 264.8 nats behind length_normalised_chance_vs_motif (10.6× dse)

**Hypothesis:** A quantitative operationalization of Kahneman & Tversky (1972): local balance is averaged across the whole sequence and sliding-window scales 2--4, while irregularity combines an over-alternating prototype with a periodic-template penalty. This is theory-inspired because K&T did not publish a unique quantitative scoring equation.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/local_representativeness.py`

### designed_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 283.9 nats behind length_normalised_chance_vs_motif (6.6× dse)

**Hypothesis:** People judge randomness by exemplar similarity: they carry a few remembered examples of obviously "designed" coin sequences — a streak of one side (HHHH…, TTTT…), perfect alternation (HTHT…, THTH…), and short repeated motifs (HHT…, HTT…, HHTT… repeated) — and a sequence looks random to the extent it is dissimilar from all of them, with similarity falling off exponentially with the number of flips at which the sequence differs from each remembered example (summed over examples, so the nearest ones dominate). The sequence that resembles the designed exemplars less is chosen as more random; how memorable each kind of exemplar is and how steeply similarity falls with mismatches are fitted.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/designed_exemplar_similarity.py`

### block_frequency_equidistribution — pruned (experiment1 end of experiment)

**Margin:** 294.3 nats behind length_normalised_chance_vs_motif (7.2× dse)

**Hypothesis:** People judge randomness by checking whether every short pattern turns up about equally often: they tally the single flips (H, T), the overlapping pairs (HH, HT, TH, TT) and the overlapping triples within a sequence, and a sequence looks random to the extent these tallies are evenly spread (high block entropy relative to the most even spread the sequence's length allows), with shorter blocks weighing more than longer ones by a fitted decay. Streaks fail because one pair/triple dominates, perfect alternation fails only on pairs and triples (it still has perfectly balanced H/T), and lopsided head counts fail at every block size.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/block_frequency_equidistribution.py`

### motif_stack — pruned (experiment1 end of experiment)

**Margin:** 307.4 nats behind length_normalised_chance_vs_motif (10.4× dse)

**Hypothesis:** Randomness = log-likelihood ratio of a fair coin versus Griffiths et al. (2018)'s four-motif stack automaton: a row-normalised six-state motif process augmented with mirror symmetry, complement symmetry, and duplication production methods, using the paper's max-path/max-method definition. Comparisons are restricted to equal-length sequences.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/motif_stack.py`

### tally_excursion_goldilocks — pruned (experiment1 end of experiment)

**Margin:** 350.7 nats behind length_normalised_chance_vs_motif (9.6× dse)

**Hypothesis:** People judge randomness by keeping a running tally of heads minus tails as they read a sequence, and they expect a fair coin's tally to wander away from balance by a moderate, characteristic amount before drifting back. A sequence looks random to the extent its tally's typical excursion from balance (relative to how far a fair coin's tally should have strayed by each point) matches that expected wandering: a tally pinned at balance (strict alternation) looks contrived, and one that runs far to one side (streaks, lopsided stretches) looks non-random, regardless of the final count.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/tally_excursion_goldilocks.py`

### falk_konold_dp — pruned (experiment1 end of experiment)

**Margin:** 541.2 nats behind length_normalised_chance_vs_motif (15.2× dse)

**Hypothesis:** Sequences seem random to the extent they are hard to encode mentally: randomness = the Difficulty Predictor DP = pure runs + 2*alternating runs under the DP-minimising parse, unnormalised by length (Falk & Konold 1997, p. 308). No free cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/falk_konold_dp.py`

### finite_experience_occurrence — pruned (experiment1 end of experiment)

**Margin:** 742.5 nats behind length_normalised_chance_vs_motif (11.1× dse)

**Hypothesis:** A string seems random to the extent one actually encounters it when watching a fair coin for the paper's focal finite stretch: randomness = log probability of occurring at least once in 20 flips (Hahn & Warren 2009). Comparisons are restricted to equal-length sequences. Penalises long runs and perfect alternation without fitted cognitive parameters.

**Source:** `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment1/model_loop/models/pruned/finite_experience_occurrence.py`
