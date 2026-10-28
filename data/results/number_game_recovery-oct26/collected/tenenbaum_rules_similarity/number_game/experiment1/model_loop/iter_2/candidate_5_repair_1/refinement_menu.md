# Refinement menu

The models you may refine, other than the incumbent `erlang_interval_size_prior`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### erlang_interval_response_lapse — rank 1, 0.8 ± 1.2 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -1277.5

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_response_lapse.py`

### few_sample_majority_vote — rank 2, 2.5 ± 1.3 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1279.1

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/few_sample_majority_vote.py`

### recency_weighted_evidence — rank 3, 3.0 ± 1.6 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -1279.6

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, with a prior favouring moderately wide ranges), but their memory of the examples is leaky: the most recently shown example counts fully as evidence, while each earlier example counts for progressively less, as if partly forgotten. Because earlier examples carry less weight, people stay less committed to the narrow concept the whole set picks out, and the more examples came before the last one, the further their generalization departs from the ideal learner's.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_evidence.py`

### located_sized_interval_prior — rank 4, 5.3 ± 2.0 nats behind the best (2.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1281.9

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), and they keep its prior over where an interval lies (a Beta density over the interval's midpoint). The one change is that the interval prior also depends on how wide the interval is, an Erlang over width with an inferred scale taken from `erlang_interval_size_prior`. The claim is that people's prior over ranges depends on both their location and their size, so generalization beyond the examples depends both on how wide a range people expect and on which part of 1-100 the examples fall in (for example, extending a range further toward small numbers than toward large ones).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/located_sized_interval_prior.py`

### bigelow_piantadosi_interval — rank 5, 8.8 ± 3.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.4

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### tempered_evidence_interval — rank 6, 11.2 ± 3.3 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1287.8

**Hypothesis:** A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/tempered_evidence_interval.py`

### bigelow_piantadosi_independent — rank 7, 12.1 ± 5.9 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1288.7

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### noise_tolerant_concept_learning — rank 8, 13.4 ± 6.1 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1290.0

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/noise_tolerant_concept_learning.py`

### log_scale_erlang_interval_prior — rank 9, 23.0 ± 6.6 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1299.6

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target, Erlang prior over interval width with an inferred scale), but they judge how wide a range is on a compressed, logarithmic number line (by the ratio of its endpoints, Weber-like) rather than by how many numbers it holds. The one change is this log-scale width in the Erlang prior: a range of fixed span counts as narrower among large numbers than among small ones, so people generalize further from examples like 83,77 than from examples like 3,7.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/log_scale_erlang_interval_prior.py`

### minimal_consistent_concept — rank 10, 210.4 ± 22.0 nats behind the best (9.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1487.0

**Hypothesis:** People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/minimal_consistent_concept.py`

### arithmetic_step_lattice — rank 11, 221.1 ± 23.4 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1497.8

**Hypothesis:** People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/arithmetic_step_lattice.py`

### featural_exemplar_similarity — rank 12, 238.1 ± 25.2 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1514.7

**Hypothesis:** People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### weak_sampling_concept_learning — rank 13, 259.2 ± 26.3 nats behind the best (9.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1535.8

**Hypothesis:** People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_learning.py`

### primacy_weighted_evidence — no comparison row

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and ranges of 1-100 (size principle, a prior favouring moderately wide ranges), but they anchor on their first impression: the first example shown counts fully as evidence about the concept, and each later example counts for progressively less, as attention and updating decline with each new example (a primacy effect). So the concept people settle on is pulled toward what the first example suggests, and the same examples shown in a different order lead to different generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/primacy_weighted_evidence.py`

### span_scaled_hull_similarity — no comparison row

**Hypothesis:** People judge a target by its numerical closeness to the range the examples span: every target inside the examples' range counts as fully similar, and similarity outside it decays exponentially with the distance from the nearest edge of that range. The width of this generalization gradient scales with how widely the examples are spread, so tightly clustered examples give a steep fall-off at the range edge and widely spread examples a gentle one; the more similar the target, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/span_scaled_hull_similarity.py`

### noisy_criterion_threshold — no comparison row

**Hypothesis:** People do the same Bayesian concept learning as the simplest well-fitting model (42 arithmetic rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not report their posterior belief directly: they say "yes" only when their belief that the target belongs to the concept clears an internal criterion of how sure they must be, and that comparison is noisy. The criterion's level and the noise are inferred, so answers follow an S-shaped function of the posterior centred on the criterion (not necessarily at 50%), producing floors of "no" for examples and of "yes" for excluded targets in proportion to how far each belief sits from the criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/noisy_criterion_threshold.py`

### gamma_shape_interval_width_prior — no comparison row

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (inferred rule weights and rule share, size principle with 10% random examples, "yes" = posterior probability the concept contains the target), but their prior over interval width is a gamma density whose shape is inferred along with its typical width, rather than an Erlang with its shape fixed at two. The one change is this free shape: a large shape means people expect ranges of a characteristic width and sharply reject ranges much wider than the examples' span (a sharp edge at the span), while a small shape means a long-tailed preference that generalizes smoothly past the edge — addressing the critique that people's yes rate drops more sharply at the span edge than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/gamma_shape_interval_width_prior.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
