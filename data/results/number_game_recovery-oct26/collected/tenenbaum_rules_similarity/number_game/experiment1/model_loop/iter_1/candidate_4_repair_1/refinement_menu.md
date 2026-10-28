# Refinement menu

The models you may refine, other than the incumbent `erlang_interval_size_prior`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### bigelow_piantadosi_interval — rank 1, 8.8 ± 3.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1285.4

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### tempered_evidence_interval — rank 2, 11.2 ± 3.3 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1287.8

**Hypothesis:** A refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100 (inferred rule weights, a rule share, and a Beta prior over interval midpoints), but they under-weight the evidence of the examples — they treat the examples as only partly informative, raising the size-principle likelihood to an inferred power below or above one rather than using it at full strength. The one change is this evidence-tempering exponent: with weaker evidence, the posterior stays spread over broader concepts and "yes" generalizes further from the examples than strong sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/tempered_evidence_interval.py`

### bigelow_piantadosi_independent — rank 3, 12.1 ± 5.9 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1288.7

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### noise_tolerant_concept_learning — rank 4, 13.4 ± 6.1 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1290.0

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 arithmetic rules and all intervals of 1-100, weighing each by how well it explains the examples under the size principle, but the share of examples they believe are random noise (not drawn from the hidden concept) is a free quantity inferred from the data rather than fixed at 10%. A learner who expects more noise keeps concepts that miss some of the examples, and so is less certain even about targets among the examples and generalizes more loosely — which is the one change from the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/noise_tolerant_concept_learning.py`

### arithmetic_step_lattice — rank 5, 221.1 ± 23.4 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1497.8

**Hypothesis:** People read the spacing between the examples as a fixed counting step (the largest step that divides every gap between examples, e.g. 3,7 → step 4) and accept a target to the extent that it can be reached from an example by repeatedly adding or subtracting that step. Targets off this arithmetic lattice get only a baseline "yes" rate, and acceptance of on-lattice targets falls off with the number of steps needed to reach them from the nearest example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/arithmetic_step_lattice.py`

### featural_exemplar_similarity — rank 6, 238.1 ± 25.2 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1514.7

**Hypothesis:** People do not induce a rule; they judge the target by its summed similarity to the remembered examples (exemplar generalization). Similarity decays exponentially with psychological distance, which combines how far apart the two numbers are in magnitude and how many of their salient surface features (parity, final digit, decade, divisibility by 3 and 5, primality, squareness, power of two) they fail to share, and the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/featural_exemplar_similarity.py`

### weak_sampling_concept_learning — rank 7, 259.2 ± 26.3 nats behind the best (9.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1535.8

**Hypothesis:** People generalize by Bayesian concept learning over the same rules and intervals, but they treat the examples only as consistent with the program's concept (weak sampling), not as drawn from it, so a large concept is not penalized for making the examples a coincidence. A concept's support is its prior times how many examples it fails to contain, with no size principle. This disagrees most sharply with the best model on sets of several examples that fit a narrow rule (e.g. 4,16,12 or 10,50,100) and on far targets: it predicts broad "yes" to targets in wide intervals and larger rules (even numbers, multiples of 2) instead of tightening around the examples as more of them arrive.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_learning.py`

### few_sample_majority_vote — no comparison row

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numeric ranges (as in the best model, with an Erlang width prior on ranges), but they cannot compute the full posterior: they bring to mind a small number of concepts, each sampled in proportion to its posterior probability, and answer "yes" if most of the sampled concepts contain the target (ties broken at random). How many concepts a person samples varies from trial to trial with an inferred mean; sampling only one reproduces the ideal posterior predictive, while sampling several makes answers more all-or-none than the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/few_sample_majority_vote.py`

### minimal_consistent_concept — no comparison row

**Hypothesis:** People do not average over many candidate concepts; they commit to the single smallest concept that contains every example — the tightest arithmetic rule (powers, multiples, final digit, contained digit, primes, even, odd, squares, cubes) or, if no rule is tighter, the exact range spanned by the examples — and say "yes" to a target essentially only if it belongs to that one concept. Apart from a constant rate of lapses in either direction, how far an excluded target lies from the examples does not matter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/minimal_consistent_concept.py`

### erlang_interval_response_lapse — no comparison row

**Hypothesis:** Refinement of erlang_interval_size_prior: people do the same Bayesian concept learning (42 rules plus every interval of 1-100 with an Erlang prior over interval width, size principle, 10% random examples), but they do not always answer from that posterior — on an inferred share of trials they lapse and give a guessed answer, saying "yes" at their own inferred base rate. The one change is this response-lapse stage, which keeps a floor of "yes" for targets the posterior rules out and of "no" for targets it is sure of, instead of answering exactly at the posterior probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_response_lapse.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
