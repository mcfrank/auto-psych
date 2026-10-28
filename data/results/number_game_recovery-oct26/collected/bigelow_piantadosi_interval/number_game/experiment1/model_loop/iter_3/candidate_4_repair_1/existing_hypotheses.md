# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## tempered_evidence_decision_gain  — rank 0, the best model on this data, ELPD-LOO -791.2

Refinement of decision_gain_concept_learning: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples) and turn their belief into a yes/no answer through the same inferred decision gain on the log-odds, but they do not weigh the examples as fully independent draws from the concept — the one change, taken from the incumbent, is that each example's evidence counts for an inferred fraction (or multiple) of a full draw. Tempering thus decides which concepts people entertain (how broadly they generalize), while the decision gain alone decides how they hedge between belief and answer, with no flat guessing floor: targets with near-zero belief still get near-zero yeses, unlike under the incumbent's guessing.

## win_stay_lose_shift_concept_tracking  — rank 1, 9.7 ± 10.1 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -800.9

People do not hold a full posterior over concepts; they hold a single running guess that they revise example by example, in the order shown (win-stay, lose-shift). After the first example they pick a concept in proportion to its Bayesian plausibility (weighted rules and intervals, size principle), keep it as long as each new example fits it, and only when an example contradicts it do they switch to a fresh concept drawn from the posterior given all examples seen so far — so broad concepts adopted early survive later examples that a full Bayesian learner would use to narrow them, and the same numbers in a different order generalize differently.

## redundant_examples_tempered_concepts  — rank 2, 11.4 ± 4.7 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -802.7

Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence and occasional guessing, but they treat additional examples as partly redundant rather than as independent draws, so the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it. The one change is this diminishing (or accelerating) return to set size: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen it much less than strong sampling predicts; an exponent of one recovers the incumbent.

## tempered_evidence_concept_guessing  — rank 3, 13.8 ± 4.9 nats behind the best (2.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -805.0

Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with the same occasional guessing, but they do not weigh the examples as fully independent random draws from the concept — each example's evidence counts for an inferred fraction (or multiple) of a full draw. The one change is this tempered size principle: weighting the evidence below one keeps broad rules and wide intervals plausible and widens generalization, while weighting it above one makes people lock onto the tightest concept consistent with the examples.

## decision_gain_concept_learning  — rank 4, 36.1 ± 10.0 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -827.4

Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not report their belief that the target is in the concept directly — they turn it into a yes/no decision with an inferred decision gain on the odds. The one change is this gain: above one, people maximize (beliefs above even odds become near-certain yeses, below even odds near-certain noes); below one, they hedge, so weak beliefs (targets just outside plausible concepts) get more yeses and strong beliefs (the examples themselves) fewer, symmetrically in log-odds rather than as a flat guessing floor.

## inferred_example_noise_belief  — rank 5, 37.3 ± 10.5 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -828.5

People do Bayesian concept learning over weighted mathematical rules and number intervals with the size principle, but they distort one assumption of the normative learner: how often they believe an example is a random number not produced by the concept. That believed noise rate is a free, inferred quantity rather than a fixed small value; the more examples people discount as noise, the more readily they accept rules that fit only some of the examples (e.g. "ends in 2" after 92,82,27,42) and the broader the intervals they entertain.

## example_noise_concept_learning  — rank 6, 38.1 ± 10.7 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -829.4

Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle) with occasional guessing, but how readily they treat an example as a stray "noise" number that does not belong to the concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise rate: a high rate lets a rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42) and a low rate forces a concept that covers every example, so it changes which concept people infer from sets containing an outlier.

## concept_learning_with_guessing  — rank 7, 39.5 ± 11.4 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -830.8

Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over 42 weighted rules and all intervals (size principle, 10% noisy examples), but on a fraction of trials they do not consult the inferred concept at all and instead guess, answering "yes" with their own guessing bias. The one change is this inferred guessing (lapse) rate with a yes-bias, which puts a floor under "yes" for targets far outside every plausible concept (the far-outside-range yeses the incumbent under-produces) and a ceiling below 1 for the examples themselves.

## serial_position_weighted_concept_learning  — rank 8, 40.0 ± 12.7 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -831.2

People read the examples one at a time and integrate them into the same Bayesian concept learning over rules and intervals (with occasional guessing) as the best model, but their memory and attention are unequal across the sequence: each example's evidence is weighted by its serial position, with a single inferred gradient that favours the most recent examples (leaky memory / forgetting) or the earliest ones (primacy, fading attention). Because earlier and later examples then count differently, generalization from multi-example sets depends on the order in which the same numbers were shown, unlike the order-blind incumbent; a flat gradient recovers it.

## response_bias_concept_learning  — rank 9, 41.4 ± 12.6 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -832.7

People do the same Bayesian concept learning as the simplest rule-and-interval model (42 weighted rules and all intervals of 1-100, size principle, 10% noisy examples), but they do not report their belief directly: they compare the evidence for "the program makes this number" against a shifted decision criterion, a response bias that adds a constant amount to the log-odds of every belief. A liberal criterion raises yeses most for middling and weak beliefs (targets just outside plausible concepts) while leaving near-certain beliefs nearly unchanged; unlike a guessing floor it does not lower yeses to the examples themselves, and unlike a decision gain it shifts all beliefs in one direction rather than pushing them apart or together.

## bigelow_piantadosi_independent  — rank 10, 44.5 ± 12.3 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -835.8

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## interval_width_power_prior  — rank 11, 46.9 ± 13.4 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -838.2

Refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 42 weighted rules and all intervals of 1-100 with the size principle, but their prior over intervals is not flat — it rises or falls as a power of the interval's width, with the exponent inferred. A prior that favours wide intervals offsets the size principle and makes people generalize further beyond the range of the examples (the incumbent under-produces "yes" to targets far outside that range); an exponent of zero recovers the incumbent.

## weak_sampling_concept_elimination  — rank 12, 81.8 ± 18.9 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -873.0

People generalize by weak-sampling hypothesis elimination: they entertain the same kinds of candidate concepts (mathematical rules and numeric intervals) but treat the examples only as positive labels on numbers that happened to come up, not as draws from the concept, so every concept consistent with the examples stays equally plausible regardless of its size (no size principle). "Yes" is the prior-weighted share of the still-consistent concepts that contain the target, with an occasional example tolerated as an error; small, tight concepts get no advantage, so generalization is far broader than under strong sampling.

## tempered_size_principle_rules_intervals  — rank 13, 203.0 ± 28.6 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -994.2

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over mathematical rules and intervals as in that model, but they discount the evidence of the examples — each example counts for only a fraction (or a multiple) of a fully independent random draw from the concept, by a fitted evidence weight — so the size principle is tempered and generalization stays broader than strong sampling predicts, reaching targets well outside the examples' range. The single change is this fitted tempering of the likelihood (the strength of the size principle), which the original fixes at full strength.

## shared_property_match_fraction  — rank 14, 339.6 ± 38.1 nats behind the best (8.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1130.8

People do not weigh competing concepts or the size of each concept; instead they note every property that all the examples have in common (parity, final digit, tens decade, being a multiple, power, square, cube or prime, lying within the examples' range) and endorse a target in proportion to how many of those shared properties it also has. The more of the examples' common properties the target matches, the more likely a "yes", regardless of how many numbers in 1-100 share those properties.

## featural_exemplar_similarity  — rank 15, 341.4 ± 29.8 nats behind the best (11.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1132.6

People do not entertain candidate rules at all; instead they judge a target by its featural similarity to each example they saw, treating numbers as bundles of salient properties (parity, divisibility by 3, 4, 5 and 10, last digit, decade, being prime, square or a power of two). Each example's similarity falls off exponentially with the number of properties on which it and the target disagree, and the chance of saying "yes" rises with the average similarity across the examples (an exemplar, GCM-style judgment with no size principle).

## tenenbaum_rules_similarity  — rank 16, 349.4 ± 58.7 nats behind the best (5.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1140.6

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## digit_grid_exemplar_similarity  — rank 17, 431.6 ± 34.5 nats behind the best (12.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1222.9

People judge a target by its similarity to the examples in the written-digit form of the numbers: each number is a point on a grid of its tens digit and its ones digit, and a target's similarity to an example falls off exponentially with the attention-weighted city-block distance between them on that grid (with an inferred share of attention given to the ones versus the tens digit). The more similar the target is on average to the examples, the more likely a "yes" — so a target sharing the examples' final digit (e.g. 52 after 92, 82, 42) or decade looks similar even when it is far away numerically, without any rule being entertained.

## log_magnitude_exemplar_similarity  — rank 18, 598.7 ± 41.3 nats behind the best (14.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1389.9

People ignore arithmetic rules entirely and generalize by exemplar similarity on a compressed (logarithmic) mental number line: the chance of saying "yes" falls off smoothly with the target's distance to the nearest example, and the falloff is wider when the examples themselves are spread far apart. This disagrees most sharply with the rule-based best model on rule-consistent targets far from every example (e.g. 64 after 16,32,2,8 — rules say yes, similarity says unlikely) and on rule-violating targets nestled among the examples (e.g. 20 after 16,32,2,8 — rules say no, similarity says likely).

## offset_progression_concepts  — no comparison row

People do the same Bayesian concept learning as the best model (weighted mathematical rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain turning belief into a yes/no answer), but their space of rules also contains offset arithmetic progressions — "every k-th number starting from r" (numbers leaving remainder r when divided by k, for steps 3 to 12), such as 1, 7, 13, 19, ... — and not only multiples counted from zero. These progressions share an inferred, a-priori small slice of prior belief, equally divided among them, so a set whose examples happen to fall on a common step (e.g. 85, 19, 91 all one more than a multiple of 6) generalizes to the other members of that progression.

## exemplar_averaged_generalization  — no comparison row

People do not infer one concept from the whole set of examples; instead each example is stored as a separate exemplar, and the target is judged by how strongly each single stored example on its own generalizes to it — through the familiar number categories (mathematical rules and numeric ranges) that the example and target both belong to, with smaller categories counting more (the size principle applied to one example at a time). These single-example generalization strengths are averaged across the examples, GCM-style, and the average belief is turned into a yes/no answer, so adding consistent examples never sharpens generalization the way joint Bayesian inference does: a target that only some examples point to (e.g. a multiple of 4 after 16, 8, 2, 64) still gets substantial support.

## predictive_representativeness_judgment  — no comparison row

People do not ask whether the target belongs to the concept they have inferred; they ask how representative the target would be as the program's next output — how much more likely the program is to produce exactly this number than a number picked at random, given the concepts the examples suggest (weighted rules and intervals, size principle). Their "yes" rises smoothly with this predicted probability of the target being generated next, so a target in a small, tight concept (32 after 16, 8, 64) is endorsed more readily than an equally certain member of a large concept (a number after examples suggesting "even" or a wide interval), unlike a membership judgment that treats all members of a believed concept alike.

## criterion_shifted_tempered_decision  — no comparison row

Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, examples' evidence tempered by an inferred weight) and pass their belief through the same inferred decision gain on its log-odds, but their yes/no criterion is not centred on even odds. The one change, taken from response_bias_concept_learning, is an inferred response bias that shifts the decision criterion by a constant on the log-odds scale, so the gain decides how sharply people separate strong from weak beliefs while the bias decides whether they lean toward "yes" or "no" — e.g. a liberal criterion gives more yeses to targets with middling or weak belief without making yeses to the examples themselves less likely.
