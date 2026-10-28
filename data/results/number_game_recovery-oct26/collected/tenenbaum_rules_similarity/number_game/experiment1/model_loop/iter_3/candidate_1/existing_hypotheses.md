# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## decade_bounded_interval_prior  — rank 0, the best model on this data, ELPD-LOO -1276.6

Refinement of `located_sized_interval_prior`: people do Bayesian concept learning over the same 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but their prior over where a range lies is not a smooth density over its midpoint: it favours ranges bounded by round decade numbers ("the 20s", "1 to 50", "51 to 100") by an inferred factor. The one change is replacing the Beta-over-midpoint location prior with this round-boundary preference, so generalization from examples tends to stop at decade boundaries — "yes" stays high up to the edge of the decade the examples fall in and drops sharply past it, a sharper edge near the examples' span than a width-only prior predicts.

## erlang_interval_size_prior  — rank 1, 0.0 ± 0.5 nats behind the best (0.0× dse: statistically tied with the best), ELPD-LOO -1276.6

Refinement of bigelow_piantadosi_interval: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (size principle, 10% of examples random, "yes" = posterior probability the concept contains the target), but their prior over intervals depends on how wide an interval is rather than where it lies. Following Tenenbaum's original number game, interval width has an Erlang prior with an inferred scale, so people a priori favour moderately sized ranges and disfavour both tiny and near-total ranges, which controls how far they generalize beyond the examples' span.

## endpoint_marking_interval_sampling  — rank 2, 0.4 ± 0.9 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -1277.0

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (Erlang prior over interval width, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but when two or more distinct examples are shown they assume the program's examples tend to mark the ends of its range (edge-marking, pedagogical sampling). The one change is an inferred evidence bonus for each end of a candidate range that coincides with the smallest or largest example, so the range that exactly spans the examples is favoured over wider ones, and people say "yes" inside the examples' span but sharply less just beyond it.

## erlang_interval_response_lapse  — rank 3, 0.8 ± 1.4 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -1277.5

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

## gamma_shape_interval_width_prior  — rank 4, 2.1 ± 1.0 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1278.7

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

## primacy_weighted_evidence  — rank 5, 2.3 ± 2.2 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1278.9

People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

## few_sample_majority_vote  — rank 6, 2.5 ± 1.4 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1279.1

People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

## recency_weighted_evidence  — rank 7, 3.0 ± 1.7 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1279.6

People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

## located_sized_interval_prior  — rank 8, 5.3 ± 2.3 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1281.9

Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

## noisy_criterion_threshold  — rank 9, 8.1 ± 3.2 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1284.7

People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

## bigelow_piantadosi_interval  — rank 10, 8.8 ± 3.3 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.4

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group. An interval's
prior follows a Beta density over its midpoint, with both shapes
inferred, so concepts in some parts of 1-100 are favoured. Examples are
sampled from the concept (the size principle), 10% of them uniformly from
1-100, and "yes" is the posterior probability that the concept contains
the target.

## tempered_evidence_interval  — rank 11, 11.2 ± 3.6 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1287.8

A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

## bigelow_piantadosi_independent  — rank 12, 12.1 ± 5.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1288.7

Bayesian concept learning over 42 rules (powers of 2-10, multiples of
3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares,
cubes) and every interval of 1-100, with a prior weight inferred for each
rule separately and a share p_mix for the rules as a group; intervals are
equally likely. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## noise_tolerant_concept_learning  — rank 13, 13.4 ± 6.0 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1290.0

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

## log_scale_erlang_interval_prior  — rank 14, 23.0 ± 6.5 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1299.6

Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

## minimal_consistent_concept  — rank 15, 210.4 ± 22.0 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1487.0

People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

## arithmetic_step_lattice  — rank 16, 221.2 ± 23.4 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1497.8

People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

## featural_exemplar_similarity  — rank 17, 238.1 ± 25.2 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1514.7

People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

## span_scaled_hull_similarity  — rank 18, 252.8 ± 25.2 nats behind the best (10.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1529.4

People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

## weak_sampling_concept_learning  — rank 19, 259.2 ± 26.4 nats behind the best (9.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1535.8

People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.
