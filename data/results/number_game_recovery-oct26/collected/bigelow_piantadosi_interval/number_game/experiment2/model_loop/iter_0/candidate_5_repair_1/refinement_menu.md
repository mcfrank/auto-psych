# Refinement menu

The models you may refine, other than the incumbent `split_rule_range_tempering`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### criterion_shifted_tempered_decision — rank 1, 4.2 ± 4.7 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -1975.7

**Hypothesis:** Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, examples' evidence tempered by an inferred weight) and pass their belief through the same inferred decision gain on its log-odds, but their yes/no criterion is not centred on even odds. The one change, taken from response_bias_concept_learning, is an inferred response bias that shifts the decision criterion by a constant on the log-odds scale, so the gain decides how sharply people separate strong from weak beliefs while the bias decides whether they lean toward "yes" or "no" — e.g. a liberal criterion gives more yeses to targets with middling or weak belief without making yeses to the examples themselves less likely.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/criterion_shifted_tempered_decision.py`

### round_landmark_interval_prior — rank 2, 5.6 ± 4.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1977.2

**Hypothesis:** People do the same Bayesian concept learning over weighted mathematical rules and numeric ranges as the best model (size principle with tempered evidence, a decision gain and criterion turning belief into a yes/no answer), but the ranges they entertain are not all equally natural: ranges whose boundaries fall on round decade landmarks (such as 1-50, 20-29, 61-100 or "the 80s") get extra prior belief, by an inferred factor, over ranges with arbitrary edges like 23-87. Generalization from examples therefore tends to stop at, or extend out to, the nearest round decade boundary rather than tracking the examples' exact span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/round_landmark_interval_prior.py`

### stochastic_example_forgetting — rank 3, 6.4 ± 4.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1978.0

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain and criterion turning belief into a yes/no answer), but their memory of the examples is unreliable: by the time they judge a target, each example has been retained only with an inferred probability, and they infer the concept from whichever examples they still hold. Their belief is therefore an average over the possible remembered subsets of the set, so larger sets are sometimes judged as if from fewer examples (broader generalization) and an odd-one-out example is sometimes forgotten, letting the tight rule the other examples share (e.g. "ends in 2" after 92, 82, 27, 42) survive without being explained away as noise.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/stochastic_example_forgetting.py`

### range_bounded_rule_concepts — rank 4, 6.9 ± 4.9 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1978.5

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 weighted rules and all intervals of 1-100, size principle with tempered evidence, a decision gain and response bias turning belief into a yes/no answer), but their rules can also be bounded to a range: besides "multiples of 10" they entertain "multiples of 10 from 70 to 100", i.e. any unbroken run of a rule's members, which takes an inferred share of each rule's prior belief. Because a range-bounded rule is smaller, the size principle favours it whenever the examples cluster, so this model disagrees most sharply with the best model on rule-consistent targets far outside the examples' range (e.g. 10 after 80, 70, 90, 100, or 6 after 96, 86, 36, 66), which it predicts far fewer yeses to, while keeping rule-violating targets inside the range at near-zero.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/range_bounded_rule_concepts.py`

### criterion_shifted_redundant_examples — rank 5, 22.7 ± 8.1 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1994.2

**Hypothesis:** Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same decision gain and shifted (biased) criterion on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_decision_gain, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it, so with sublinear growth one example already supports a fairly tight concept while three or four sharpen generalization less than strong sampling predicts, while the criterion shift separately sets the overall lean toward "yes" (an exponent of one recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/criterion_shifted_redundant_examples.py`

### redundant_examples_decision_gain — rank 6, 34.2 ± 10.8 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2005.7

**Hypothesis:** Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same inferred decision gain on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_tempered_concepts, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen generalization much less than strong sampling predicts (an exponent of one recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/redundant_examples_decision_gain.py`

### win_stay_lose_shift_criterion_shift — rank 7, 77.4 ± 15.9 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2048.9

**Hypothesis:** Refinement of win_stay_lose_shift_decision_gain: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it) and pass their belief that this tracked concept contains the target through an inferred decision gain on its log-odds. The one change, taken from the incumbent criterion_shifted_tempered_decision, is an inferred response bias that shifts the yes/no criterion away from even odds by a constant on the log-odds scale, so a liberal (or conservative) criterion raises (or lowers) yeses to targets with middling or weak belief under the order-dependent guess without much changing yeses to the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/win_stay_lose_shift_criterion_shift.py`

### belief_carryover_assimilation — no comparison row

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 weighted rules and all intervals of 1-100, size principle with evidence tempered separately for rules and ranges, a decision gain and criterion shift turning belief into a yes/no answer), but they do not fully reset their belief between trials: the belief they act on is partly carried over from the previous trial, so their confidence that the current target belongs to the concept is pulled, by an inferred inertia weight, toward the confidence they had about the previous trial's target. This sequential assimilation makes consecutive answers in a session more alike than the stimuli alone warrant (e.g. a "no"-worthy target asked right after a near-certain "yes" gets more yeses); zero inertia recovers the best model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/belief_carryover_assimilation.py`

### disjunctive_sampled_concepts — no comparison row

**Hypothesis:** People do not consult a full, graded belief over all concepts; instead they bring to mind a small handful of candidate concepts (mathematical rules or numeric ranges), each recalled in proportion to how plausible it is given the examples (size principle, tempered evidence), and answer "yes" if the target fits any one of the concepts they recalled, occasionally answering at random instead. Because the check is disjunctive, a target supported by even a modestly plausible concept gets many yeses (the more concepts recalled, the more), while the yes rate saturates for well-supported targets — an asymmetric belief-to-answer mapping rather than a symmetric gain and criterion shift on the log-odds.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/disjunctive_sampled_concepts.py`

### first_example_anchored_concepts — no comparison row

**Hypothesis:** People infer the program's concept by the same Bayesian weighing of mathematical rules and numeric ranges as the best model, but the first example they see anchors that inference: they treat it as the defining instance of the program, so every concept containing the first example gets extra belief, while concepts that would have to dismiss it as a stray number are disfavoured more than concepts that dismiss a later example. The same numbers shown in a different order therefore generalize differently — an odd-one-out shown first pulls belief toward concepts that include it, while the same odd-one-out shown last is readily written off.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/first_example_anchored_concepts.py`

## Pruned models (out of the set; narrowest margin first)

### win_stay_lose_shift_decision_gain — pruned (experiment1 end of experiment)

**Margin:** 9.6 nats behind criterion_shifted_redundant_examples; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 9

**Hypothesis:** Refinement of win_stay_lose_shift_concept_tracking: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it), but they do not answer by occasionally guessing at random. The one change, taken from the incumbent, replaces the flat guessing floor with an inferred decision gain on the log-odds of their belief that the tracked concept contains the target: people hedge (or sharpen) symmetrically between belief and answer, so targets with near-zero belief under the order-dependent guess still get near-zero yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_decision_gain.py`

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 12.3 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted mathematical rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain turning belief into a yes/no answer), but their space of rules also contains offset arithmetic progressions — "every k-th number starting from r" (numbers leaving remainder r when divided by k, for steps 3 to 12), such as 1, 7, 13, 19, ... — and not only multiples counted from zero. These progressions share an inferred, a-priori small slice of prior belief, equally divided among them, so a set whose examples happen to fall on a common step (e.g. 85, 19, 91 all one more than a multiple of 6) generalizes to the other members of that progression.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### tempered_evidence_decision_gain — pruned (experiment1 end of experiment)

**Margin:** 13.6 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** Refinement of decision_gain_concept_learning: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples) and turn their belief into a yes/no answer through the same inferred decision gain on the log-odds, but they do not weigh the examples as fully independent draws from the concept — the one change, taken from the incumbent, is that each example's evidence counts for an inferred fraction (or multiple) of a full draw. Tempering thus decides which concepts people entertain (how broadly they generalize), while the decision gain alone decides how they hedge between belief and answer, with no flat guessing floor: targets with near-zero belief still get near-zero yeses, unlike under the incumbent's guessing.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_decision_gain.py`

### win_stay_lose_shift_concept_tracking — pruned (experiment1 end of experiment)

**Margin:** 23.3 nats behind criterion_shifted_redundant_examples (2.7× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they hold a single running guess that they revise example by example, in the order shown (win-stay, lose-shift). After the first example they pick a concept in proportion to its Bayesian plausibility (weighted rules and intervals, size principle), keep it as long as each new example fits it, and only when an example contradicts it do they switch to a fresh concept drawn from the posterior given all examples seen so far — so broad concepts adopted early survive later examples that a full Bayesian learner would use to narrow them, and the same numbers in a different order generalize differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concept_tracking.py`

### redundant_examples_tempered_concepts — pruned (experiment1 end of experiment)

**Margin:** 25.0 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence and occasional guessing, but they treat additional examples as partly redundant rather than as independent draws, so the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it. The one change is this diminishing (or accelerating) return to set size: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen it much less than strong sampling predicts; an exponent of one recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/redundant_examples_tempered_concepts.py`

### tempered_evidence_concept_guessing — pruned (experiment1 end of experiment)

**Margin:** 27.4 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with the same occasional guessing, but they do not weigh the examples as fully independent random draws from the concept — each example's evidence counts for an inferred fraction (or multiple) of a full draw. The one change is this tempered size principle: weighting the evidence below one keeps broad rules and wide intervals plausible and widens generalization, while weighting it above one makes people lock onto the tightest concept consistent with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_concept_guessing.py`

### decision_gain_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 49.7 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not report their belief that the target is in the concept directly — they turn it into a yes/no decision with an inferred decision gain on the odds. The one change is this gain: above one, people maximize (beliefs above even odds become near-certain yeses, below even odds near-certain noes); below one, they hedge, so weak beliefs (targets just outside plausible concepts) get more yeses and strong beliefs (the examples themselves) fewer, symmetrically in log-odds rather than as a flat guessing floor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decision_gain_concept_learning.py`

### inferred_example_noise_belief — pruned (experiment1 end of experiment)

**Margin:** 50.9 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** People do Bayesian concept learning over weighted mathematical rules and number intervals with the size principle, but they distort one assumption of the normative learner: how often they believe an example is a random number not produced by the concept. That believed noise rate is a free, inferred quantity rather than a fixed small value; the more examples people discount as noise, the more readily they accept rules that fit only some of the examples (e.g. "ends in 2" after 92,82,27,42) and the broader the intervals they entertain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_example_noise_belief.py`

### example_noise_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 51.7 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle) with occasional guessing, but how readily they treat an example as a stray "noise" number that does not belong to the concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise rate: a high rate lets a rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42) and a low rate forces a concept that covers every example, so it changes which concept people infer from sets containing an outlier.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/example_noise_concept_learning.py`

### concept_learning_with_guessing — pruned (experiment1 end of experiment)

**Margin:** 53.1 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over 42 weighted rules and all intervals (size principle, 10% noisy examples), but on a fraction of trials they do not consult the inferred concept at all and instead guess, answering "yes" with their own guessing bias. The one change is this inferred guessing (lapse) rate with a yes-bias, which puts a floor under "yes" for targets far outside every plausible concept (the far-outside-range yeses the incumbent under-produces) and a ceiling below 1 for the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/concept_learning_with_guessing.py`

### serial_position_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 53.6 nats behind criterion_shifted_redundant_examples (3.2× dse)

**Hypothesis:** People read the examples one at a time and integrate them into the same Bayesian concept learning over rules and intervals (with occasional guessing) as the best model, but their memory and attention are unequal across the sequence: each example's evidence is weighted by its serial position, with a single inferred gradient that favours the most recent examples (leaky memory / forgetting) or the earliest ones (primacy, fading attention). Because earlier and later examples then count differently, generalization from multi-example sets depends on the order in which the same numbers were shown, unlike the order-blind incumbent; a flat gradient recovers it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_concept_learning.py`

### response_bias_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 55.0 nats behind criterion_shifted_redundant_examples (3.6× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest rule-and-interval model (42 weighted rules and all intervals of 1-100, size principle, 10% noisy examples), but they do not report their belief directly: they compare the evidence for "the program makes this number" against a shifted decision criterion, a response bias that adds a constant amount to the log-odds of every belief. A liberal criterion raises yeses most for middling and weak beliefs (targets just outside plausible concepts) while leaving near-certain beliefs nearly unchanged; unlike a guessing floor it does not lower yeses to the examples themselves, and unlike a decision gain it shifts all beliefs in one direction rather than pushing them apart or together.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/response_bias_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 58.1 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### interval_width_power_prior — pruned (experiment1 end of experiment)

**Margin:** 60.5 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 42 weighted rules and all intervals of 1-100 with the size principle, but their prior over intervals is not flat — it rises or falls as a power of the interval's width, with the exponent inferred. A prior that favours wide intervals offsets the size principle and makes people generalize further beyond the range of the examples (the incumbent under-produces "yes" to targets far outside that range); an exponent of zero recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_power_prior.py`

### weak_sampling_concept_elimination — pruned (experiment1 end of experiment)

**Margin:** 95.3 nats behind criterion_shifted_redundant_examples (5.5× dse)

**Hypothesis:** People generalize by weak-sampling hypothesis elimination: they entertain the same kinds of candidate concepts (mathematical rules and numeric intervals) but treat the examples only as positive labels on numbers that happened to come up, not as draws from the concept, so every concept consistent with the examples stays equally plausible regardless of its size (no size principle). "Yes" is the prior-weighted share of the still-consistent concepts that contain the target, with an occasional example tolerated as an error; small, tight concepts get no advantage, so generalization is far broader than under strong sampling.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_elimination.py`

### predictive_representativeness_judgment — pruned (experiment1 end of experiment)

**Margin:** 96.9 nats behind criterion_shifted_redundant_examples (4.5× dse)

**Hypothesis:** People do not ask whether the target belongs to the concept they have inferred; they ask how representative the target would be as the program's next output — how much more likely the program is to produce exactly this number than a number picked at random, given the concepts the examples suggest (weighted rules and intervals, size principle). Their "yes" rises smoothly with this predicted probability of the target being generated next, so a target in a small, tight concept (32 after 16, 8, 64) is endorsed more readily than an equally certain member of a large concept (a number after examples suggesting "even" or a wide interval), unlike a membership judgment that treats all members of a believed concept alike.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/predictive_representativeness_judgment.py`

### tempered_size_principle_rules_intervals — pruned (experiment1 end of experiment)

**Margin:** 216.6 nats behind criterion_shifted_redundant_examples (6.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over mathematical rules and intervals as in that model, but they discount the evidence of the examples — each example counts for only a fraction (or a multiple) of a fully independent random draw from the concept, by a fitted evidence weight — so the size principle is tempered and generalization stays broader than strong sampling predicts, reaching targets well outside the examples' range. The single change is this fitted tempering of the likelihood (the strength of the size principle), which the original fixes at full strength.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_rules_intervals.py`

### exemplar_averaged_generalization — pruned (experiment1 end of experiment)

**Margin:** 285.1 nats behind criterion_shifted_redundant_examples (8.7× dse)

**Hypothesis:** People do not infer one concept from the whole set of examples; instead each example is stored as a separate exemplar, and the target is judged by how strongly each single stored example on its own generalizes to it — through the familiar number categories (mathematical rules and numeric ranges) that the example and target both belong to, with smaller categories counting more (the size principle applied to one example at a time). These single-example generalization strengths are averaged across the examples, GCM-style, and the average belief is turned into a yes/no answer, so adding consistent examples never sharpens generalization the way joint Bayesian inference does: a target that only some examples point to (e.g. a multiple of 4 after 16, 8, 2, 64) still gets substantial support.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_averaged_generalization.py`

### shared_property_match_fraction — pruned (experiment1 end of experiment)

**Margin:** 353.2 nats behind criterion_shifted_redundant_examples (9.4× dse)

**Hypothesis:** People do not weigh competing concepts or the size of each concept; instead they note every property that all the examples have in common (parity, final digit, tens decade, being a multiple, power, square, cube or prime, lying within the examples' range) and endorse a target in proportion to how many of those shared properties it also has. The more of the examples' common properties the target matches, the more likely a "yes", regardless of how many numbers in 1-100 share those properties.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_match_fraction.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 354.9 nats behind criterion_shifted_redundant_examples (12.3× dse)

**Hypothesis:** People do not entertain candidate rules at all; instead they judge a target by its featural similarity to each example they saw, treating numbers as bundles of salient properties (parity, divisibility by 3, 4, 5 and 10, last digit, decade, being prime, square or a power of two). Each example's similarity falls off exponentially with the number of properties on which it and the target disagree, and the chance of saying "yes" rises with the average similarity across the examples (an exemplar, GCM-style judgment with no size principle).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 362.9 nats behind criterion_shifted_redundant_examples (5.8× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### digit_grid_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 445.2 nats behind criterion_shifted_redundant_examples (13.3× dse)

**Hypothesis:** People judge a target by its similarity to the examples in the written-digit form of the numbers: each number is a point on a grid of its tens digit and its ones digit, and a target's similarity to an example falls off exponentially with the attention-weighted city-block distance between them on that grid (with an inferred share of attention given to the ones versus the tens digit). The more similar the target is on average to the examples, the more likely a "yes" — so a target sharing the examples' final digit (e.g. 52 after 92, 82, 42) or decade looks similar even when it is far away numerically, without any rule being entertained.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_grid_exemplar_similarity.py`

### log_magnitude_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 612.2 nats behind criterion_shifted_redundant_examples (15.2× dse)

**Hypothesis:** People ignore arithmetic rules entirely and generalize by exemplar similarity on a compressed (logarithmic) mental number line: the chance of saying "yes" falls off smoothly with the target's distance to the nearest example, and the falloff is wider when the examples themselves are spread far apart. This disagrees most sharply with the rule-based best model on rule-consistent targets far from every example (e.g. 64 after 16,32,2,8 — rules say yes, similarity says unlikely) and on rule-violating targets nestled among the examples (e.g. 20 after 16,32,2,8 — rules say no, similarity says likely).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_magnitude_exemplar_similarity.py`
