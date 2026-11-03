# Refinement menu

The models you may refine, other than the incumbent `bigelow_piantadosi_independent`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### tenenbaum_rules_similarity — rank 1, 322.5 ± 41.8 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1249.1

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### exemplar_feature_similarity_gcm — no comparison row

**Hypothesis:** People do not entertain explicit rules; they judge the target by its summed psychological similarity to the remembered examples (an exemplar / generalized-context model). Similarity decays exponentially with distance in a feature space where each number is described by its magnitude and by its surface properties (parity, last digit, tens digit, divisibility by 3/4/5, being prime or a square), with attention split between magnitude and properties; the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### weak_sampling_concept_generalization — no comparison row

**Hypothesis:** People entertain the same mathematical rules and numeric ranges as candidate concepts, but treat the examples only as labelled members of the concept rather than as random draws from it (weak sampling, no size principle): every concept consistent with the examples stays credible in proportion to its prior, however broad it is, so adding examples rules concepts out but never sharpens generalization toward the narrowest one. The sharpest disagreement with the best (strong-sampling) model is on multi-example sets that fit both a narrow rule and broader ones (e.g. 16,8,64 or 83,38,58,68): this model predicts broad, graded "yes" to targets in wide consistent ranges and broad rules, where the size-principle model predicts confident yes only for the narrow rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_generalization.py`

### digit_feature_exemplar_similarity — no comparison row

**Hypothesis:** People judge a target by its featural resemblance to the examples in decimal notation: each example's similarity to the target falls off exponentially with whether the two differ in their ones digit and whether they differ in their tens digit, and the probability of saying yes grows with the target's summed (averaged) similarity to the shown examples, as in an exemplar model over digit features rather than a search over explicit rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/digit_feature_exemplar_similarity.py`

### concept_learning_with_guess_lapse — no comparison row

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules with separately weighted priors plus every interval of 1-100 (size principle, 10% of examples random), but on a fixed fraction of trials they do not consult the inferred concept at all and instead answer by guessing, saying "yes" at their own baseline rate. The one change is this response-lapse component, which lets the model explain "no" answers to targets the concept clearly contains (even the shown examples) and stray "yes" answers to targets it clearly excludes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/concept_learning_with_guess_lapse.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
