# Refinement menu

The models you may refine, other than the incumbent `coherence_tuned_attention_exemplar`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### graded_shared_factor_exemplar_similarity — rank 1, 40.8 ± 10.1 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1592.8

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, but the number-theoretic part of that similarity is graded by how large a factor the target and each example share, not by whether one divides the other. The single change replaces the all-or-none divisibility mismatch with a graded shared-factor mismatch (zero when one number divides the other, one when they share no factor, in between in proportion to the log size of their greatest common divisor relative to the smaller number), so targets that share a big factor with the examples (e.g. 42 after 12, 18, 30) are endorsed more than divisibility alone predicts, which is what the common-factor rule gap in the critique shows.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/graded_shared_factor_exemplar_similarity.py`

### recency_weighted_exemplar_similarity — rank 2, 43.0 ± 10.2 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1594.9

**Hypothesis:** People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_exemplar_similarity.py`

### common_factor_exemplar_similarity — rank 3, 44.7 ± 10.3 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1596.7

**Hypothesis:** Refinement of recency_weighted_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples (mismatches on magnitude, final digit, parity and pairwise divisibility), but when several examples share a common factor of 3 or more they also encode that shared factor as a property of the whole set, so a target that is not a multiple of it is psychologically farther from every example. The single change is this one set-level "shared common factor" mismatch dimension in the distance, which predicts more "yes" answers to multiples of the examples' greatest common divisor than pairwise exemplar similarity alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/common_factor_exemplar_similarity.py`

### tempered_concept_learning_with_guessing — rank 4, 75.6 ± 17.3 nats behind the best (4.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1627.6

**Hypothesis:** Refinement of tempered_size_principle_concept_learning: people do the same Bayesian concept learning over 42 rules and all intervals of 1-100 with a partially applied (tempered) size principle, but on some fraction of trials they do not consult the inferred concept at all and simply guess, answering "yes" with their own inferred guessing bias. The single change is this inferred lapse-to-guessing process, which predicts that even targets the concept clearly excludes get a substantial floor of "yes" answers (and clearly included targets a ceiling of "no" answers) that is the same for every set and target, rather than the near-certain answers the pure concept learner gives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/tempered_concept_learning_with_guessing.py`

### log_odds_distorted_concept_learning — rank 5, 83.5 ± 16.9 nats behind the best (4.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1635.4

**Hypothesis:** People are Bayesian concept learners over mathematical rules (powers, multiples, final and contained digits, primes, parity, squares, cubes) and numeric intervals with the size principle, but they misperceive their own posterior belief that the target belongs to the concept the way people distort probabilities generally: the belief is compressed or expanded linearly in log-odds around an inferred crossover point before it is turned into a yes/no answer. Weak beliefs are therefore over-endorsed and near-certain ones under-endorsed (or the reverse), while the ordering of targets is still set entirely by the normative posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/log_odds_distorted_concept_learning.py`

### tempered_size_principle_concept_learning — rank 6, 106.9 ± 21.5 nats behind the best (5.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1658.9

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 42 rules and all intervals of 1-100, but they apply the size principle only partially — they treat the examples as weaker evidence than a fully strong-sampling learner would, so the evidence from the examples is tempered by one inferred strength (1 recovers the incumbent; below 1 means broader generalization). The single change is this evidence-strength exponent, applied equally to every example regardless of order; it predicts more "yes" answers to targets outside the tightest consistent concept, most of all after a single example, where the incumbent generalises too narrowly.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_concept_learning.py`

### recency_forgetting_concept_learning — rank 7, 120.3 ± 22.0 nats behind the best (5.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1672.2

**Hypothesis:** People are Bayesian concept learners (rules and intervals, size principle) who reason only from the examples they still hold in memory: the last-shown example is always remembered, and each earlier example is forgotten with a probability that grows the further back it was shown, so the concept is inferred from a recency-biased subset of the set. This disagrees most with the order-invariant best model on multi-example sets whose early examples point to a different concept than the last one: for targets close to the last example but outside the concept implied by the whole set it predicts many more "yes" answers, and for targets near only the first example, fewer.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_forgetting_concept_learning.py`

### mixed_sampling_concept_learning — rank 8, 128.5 ± 24.1 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1680.4

**Hypothesis:** Refinement of bigelow_piantadosi_independent: the same Bayesian concept learner over 42 rules and all intervals of 1-100 (rule weights inferred separately, intervals equally likely, 10% noise examples), but people only partly believe the examples were sampled from the concept: each example is taken as drawn from the concept with an inferred probability (strong sampling, the size principle) and otherwise as an arbitrary number that merely happens to belong to it (weak sampling). The one change is this strong/weak sampling mixture, which weakens the size principle's penalty on large concepts, so people generalise more broadly than the incumbent predicts, most of all from a single example.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/mixed_sampling_concept_learning.py`

### recency_decayed_concept_learning — rank 9, 168.2 ± 25.3 nats behind the best (6.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1720.2

**Hypothesis:** People infer the hidden program by Bayesian concept learning over mathematical rules and numeric intervals, but their memory of the examples decays with recency: the most recently shown example counts as full evidence while each earlier example is discounted by a constant factor per position back. Concepts consistent with the later examples therefore dominate, so targets near or sharing structure with the last-shown example get more "yes" answers than those tied to the first-shown one, and evidence from long sets is weaker (broader generalization).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_decayed_concept_learning.py`

### shared_divisor_rule_abstraction — rank 10, 182.3 ± 21.4 nats behind the best (8.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1734.2

**Hypothesis:** People abstract a single explicit rule from the examples: "multiples of g", where g is the greatest common divisor the examples share. They endorse targets that are multiples of g and reject those that are not, and they trust this rule more the less likely it is that the examples would share g by coincidence (more examples and a larger g); when the examples share no factor above 1, people fall back on nothing but how close the target is to the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/shared_divisor_rule_abstraction.py`

### bigelow_piantadosi_independent — rank 11, 240.2 ± 29.8 nats behind the best (8.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1792.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### bigelow_piantadosi_interval — rank 12, 250.2 ± 33.9 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1802.1

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_interval.py`

### recency_weighted_concept_learning — rank 13, 274.0 ± 36.8 nats behind the best (7.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1826.0

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do Bayesian concept learning over the same 42 rules and 1-100 intervals (with a midpoint-dependent interval prior and the size principle), but they weigh the examples by recency — each example's evidence is discounted geometrically by how many examples were shown after it (one inferred decay rate, total evidence held fixed), so concepts consistent with the most recently shown examples gain posterior weight and targets near the last example are endorsed more. The single change is this order-dependent evidence weighting, motivated by the critique's recency discrepancy; a decay of 1 recovers the original model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_concept_learning.py`

### tenenbaum_rules_similarity — rank 14, 502.8 ± 72.5 nats behind the best (6.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2054.7

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
