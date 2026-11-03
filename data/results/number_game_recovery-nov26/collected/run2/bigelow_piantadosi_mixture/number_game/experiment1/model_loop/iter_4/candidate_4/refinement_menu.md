# Refinement menu

The models you may refine, other than the incumbent `concepts_sampled_without_replacement`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### without_replacement_size_principle — rank 1, 0.2 ± 1.2 nats behind the best (0.1× dse: statistically tied with the best), ELPD-LOO -907.4

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (equal rule prior, 10% of examples random, occasional guessing at their own yes rate), but they understand that the program shows distinct examples, so each further example is read as a draw from the concept's members not yet shown (sampling without replacement) rather than from all its members again. The one change is this without-replacement size principle, which adds no free parameter and predicts that several examples sharpen generalization toward small concepts more than the standard model does: a few examples that nearly exhaust a small rule (e.g. 16, 8, 64 for powers of 2) or a short range make that concept far more credible than broader alternatives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/without_replacement_size_principle.py`

### rule_breadth_prior_concepts — rank 2, 0.6 ± 1.1 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -907.8

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random), with occasional guessing at their own yes rate — but before seeing any examples they already prefer rules by their breadth, giving each rule a prior weight that rises or falls as a power of how many numbers it contains (a preference for specific, "interesting" rules over broad ones, or the reverse). The one change is replacing the flat prior across rules with this single size-graded rule prior, which predicts that broad rules such as "odd" or "even" win less (or more) of the posterior than an equal-prior learner gives them, e.g. fewer "yes" answers to arbitrary odd numbers after 85, 19, 91.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/rule_breadth_prior_concepts.py`

### breadth_prior_concepts_free_outlier_rate — rank 3, 0.6 ± 1.6 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -907.8

**Hypothesis:** Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules (with the same breadth-graded rule prior) and every interval of 1-100, with the size principle and occasional guessing at their own yes rate — but how often they believe the program emits a stray random example is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate (the component taken from `digit_concepts_inferred_outlier_rate`), which claims that part of what the lapse now absorbs is really people discounting a misfitting example as a stray, so tight rules survive an odd-one-out example (e.g. 85, 19, 91 or 14, 47) instead of being abandoned for broad ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/breadth_prior_concepts_free_outlier_rate.py`

### counting_by_twos_run_concepts — rank 4, 0.8 ± 1.2 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -908.0

**Hypothesis:** Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same arithmetic and digit rules (with their size-graded rule prior) and every interval of 1-100, with the size principle, 10% of examples random and occasional guessing. They also entertain one more kind of concept, "counting by twos" runs: a bounded stretch of every other number, such as 60, 62, 64, 66 or 31, 33, 35, i.e. the even or the odd numbers within a limited range. All such runs are equally likely a priori, and the family as a whole takes an inferred share of prior belief relative to the rules. This predicts that examples spaced by two (e.g. 60, 62, 64) draw "yes" to nearby same-parity numbers but not to same-parity numbers far away, a pattern that neither "even numbers" nor a numeric range gives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/counting_by_twos_run_concepts.py`

### equal_prior_concepts_with_guess_lapse — rank 5, 1.0 ± 1.9 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -908.2

**Hypothesis:** Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle and 10% of examples random, and on a fixed fraction of trials they ignore the concept and guess "yes" at their own baseline rate — but they hold no learned preference among the rules: every rule gets the same prior weight. The one change is replacing the 45 separately inferred rule weights with a single shared rule prior, which claims that the lapse model's advantage comes from occasional guessing, not from idiosyncratic rule preferences, so the model should generalize as well with 43 fewer free parameters.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/equal_prior_concepts_with_guess_lapse.py`

### pedagogical_sampling_concepts — rank 6, 1.7 ± 2.0 nats behind the best (0.9× dse: statistically tied with the best), ELPD-LOO -908.9

**Hypothesis:** People treat the examples as chosen on purpose by a helpful teacher rather than drawn at random from the hidden concept: they assume the program picks, from among the concept's members, numbers that would best lead a learner to that concept (numbers that few other plausible concepts contain), and they infer the concept by reasoning about why these particular examples were chosen. Over the same arithmetic rules, digit concepts and numeric ranges (with occasional guessing), this pedagogical reading predicts that an example typical of many concepts (e.g. a small even number) says less about the concept than a distinctive one, so generalization follows the concept the examples single out more than the size principle alone implies, with the strength of this teacher assumption inferred from the answers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/pedagogical_sampling_concepts.py`

### occam_specificity_prior_all_concepts — rank 7, 2.0 ± 1.9 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -909.3

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random, occasional guessing at their own yes rate), but they bring one general Occam-style specificity bias to every candidate concept, rules and numeric ranges alike: before seeing any examples, a concept's prior plausibility falls (or rises) as a power of how many numbers it contains. The one change is replacing the flat priors within the rules and within the intervals with this single shared size-graded prior, which predicts not only weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) but also tighter, more local "yes" ranges around clustered examples than an equal-prior learner, with both effects set by the same one parameter.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/occam_specificity_prior_all_concepts.py`

### iter2_candidate3 — rank 8, 3.7 ± 2.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -911.0

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle, 10% of examples random, and a guess lapse at their own baseline "yes" rate — but the rules are not all equally salient: each rule's prior weight is a draw from a shared population of rule saliences whose spread is inferred, so a rule departs from the common weight only as far as the answers demand. The one change is replacing the single shared rule prior with this partially pooled (hierarchical) rule prior, which claims that some rules (e.g. "odd numbers") are systematically less or more attractive than others, yet predicts generalization closer to the equal-prior model than 45 freely inferred weights would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/iter2_candidate3.py`

### soft_confidence_criterion_decision — rank 9, 7.3 ± 3.5 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -914.6

**Hypothesis:** People do Bayesian concept learning exactly as in the simplest digit-concept model (equal-prior mathematical and digit rules plus size-weighted intervals), but they do not answer "yes" with the probability that the concept contains the target; instead they compare their degree of belief against a personal criterion of how sure they must be, answering "yes" when their confidence that the target belongs exceeds that criterion, with graded (noisy) sensitivity around it. This single decision-threshold rule predicts that moderately probable targets are pushed toward "no" (or "yes") depending on where the criterion sits, and that near-certain beliefs are softened or sharpened by how steeply responses track the confidence-versus-criterion gap.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/soft_confidence_criterion_decision.py`

### digit_concepts_with_guess_lapse — rank 10, 8.0 ± 5.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -915.3

**Hypothesis:** Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning over Tenenbaum's mathematical rules, the digit concepts ("contains the digit d", "ends in 0") and size-weighted intervals exactly as in that model, but on a small fixed fraction of trials they do not consult the inferred concept and instead guess, saying "yes" at their own baseline rate. The one change is this response-lapse component (taken from `concept_learning_with_guess_lapse`), which predicts occasional "no" answers to targets the concept surely contains and occasional "yes" answers to targets it surely excludes, instead of near-certain responses there.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/digit_concepts_with_guess_lapse.py`

### tenenbaum_digit_concepts — rank 11, 11.0 ± 8.1 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -918.2

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over Tenenbaum's mathematical rules and size-weighted intervals, but they also read the examples as written numerals, so their hypothesis space additionally contains digit concepts — "contains the digit d" (for each digit 0–9) and "ends in 0" — sharing the rules' equal prior. The one change is adding these digit concepts, which predicts that examples sharing a digit (e.g. 83, 38, 58, 68) lead people to say yes to other numbers containing that digit (e.g. 84, 87), even when those numbers are far from any arithmetic rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_digit_concepts.py`

### digit_concepts_inferred_outlier_rate — rank 12, 12.3 ± 7.9 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -919.5

**Hypothesis:** Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning with the size principle over Tenenbaum's arithmetic rules, size-weighted intervals and the digit concepts ("contains the digit d", "ends in 0"), while allowing that any example may be a stray random number from 1–100 — and how often they believe the program produces such strays is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate, which predicts that people discount an example that fails to fit a tight concept as a stray rather than abandoning the concept for a broader one, so narrow rules survive a misfitting example and generalization to their members stays strong.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/digit_concepts_inferred_outlier_rate.py`

### order_weighted_concept_learning — rank 13, 13.5 ± 8.2 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -920.8

**Hypothesis:** People do Bayesian concept learning over mathematical rules, digit concepts and numeric intervals (the size principle), but they read the examples one after another and do not weigh them equally: their attention to an example shifts systematically with its position in the sequence, so later-read examples (recency) or earlier-read examples (primacy) count for more of the evidence about the concept. The one claim is this order-dependent weighting of each example's evidence — the total amount of evidence stays the same — which predicts that the same set shown in a different order generalizes more toward the concepts favoured by the emphasized examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/order_weighted_concept_learning.py`

### conservative_evidence_digit_concepts — rank 14, 15.4 ± 7.6 nats behind the best (2.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -922.6

**Hypothesis:** People do Bayesian concept learning over Tenenbaum's mathematical rules, digit concepts and size-weighted intervals, but they are conservative updaters: they underweight (or overweight) the evidence each example provides, so the likelihood of the examples under each concept enters their belief only to a fitted power rather than at full strength. This single mis-weighting predicts that generalization tightens toward the narrowest consistent concept more slowly (or faster) with added examples than an ideal size-principle learner would, so multi-example sets still yield graded "yes" to broader consistent concepts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/conservative_evidence_digit_concepts.py`

### concept_learning_with_guess_lapse — rank 15, 18.1 ± 4.2 nats behind the best (4.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -925.3

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules with separately weighted priors plus every interval of 1-100 (size principle, 10% of examples random), but on a fixed fraction of trials they do not consult the inferred concept at all and instead answer by guessing, saying "yes" at their own baseline rate. The one change is this response-lapse component, which lets the model explain "no" answers to targets the concept clearly contains (even the shown examples) and stray "yes" answers to targets it clearly excludes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/concept_learning_with_guess_lapse.py`

### bigelow_piantadosi_independent — rank 16, 19.3 ± 5.2 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -926.6

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/bigelow_piantadosi_independent.py`

### first_example_anchored_concept_learning — rank 17, 30.4 ± 10.3 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -937.6

**Hypothesis:** People do Bayesian concept learning over the same arithmetic rules, digit concepts and numeric intervals as the best model (size principle, equal rule prior, occasional guessing), but they build their guess about the concept from the first example they read and treat it as a sure member of the concept: only the examples that follow can be dismissed as stray random numbers. This primacy-of-the-anchor claim predicts that a set whose first example misfits a tight rule shared by the later ones generalizes broadly (the rule must include the anchor), whereas the same misfit shown last is discounted and the tight rule survives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/first_example_anchored_concept_learning.py`

### smallest_rule_commitment — rank 18, 197.0 ± 20.0 nats behind the best (9.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1104.2

**Hypothesis:** People do not weigh many candidate concepts at once; they search for the single most specific rule that covers every example (the smallest of the arithmetic and digit rules such as "powers of 2", "multiples of 7", "ends in 3" or "contains an 8"), commit to it, and answer yes to a target if and only if it obeys that rule, falling back to the range spanned by the examples only when no rule fits them all. Apart from response noise (a fixed rate of yes inside the chosen rule and of yes outside it), generalization is all-or-none with no graded similarity or averaging over alternative concepts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/smallest_rule_commitment.py`

### weak_sampling_concept_generalization — rank 19, 210.9 ± 21.6 nats behind the best (9.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1118.2

**Hypothesis:** People entertain the same mathematical rules and numeric ranges as candidate concepts, but treat the examples only as labelled members of the concept rather than as random draws from it (weak sampling, no size principle): every concept consistent with the examples stays credible in proportion to its prior, however broad it is, so adding examples rules concepts out but never sharpens generalization toward the narrowest one. The sharpest disagreement with the best (strong-sampling) model is on multi-example sets that fit both a narrow rule and broader ones (e.g. 16,8,64 or 83,38,58,68): this model predicts broad, graded "yes" to targets in wide consistent ranges and broad rules, where the size-principle model predicts confident yes only for the narrow rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/weak_sampling_concept_generalization.py`

### concept_mediated_exemplar_similarity — rank 20, 269.0 ± 25.4 nats behind the best (10.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1176.2

**Hypothesis:** People do not infer a single concept that covers all the examples; they store each example and judge the target by its summed (averaged) similarity to those stored examples, as in an exemplar model. The similarity of the target to one example is how likely the two are to share a number concept — an arithmetic or digit rule (e.g. "powers of 2", "contains a 3") or a numeric range — judged from that example alone, so sharing a narrow property counts for more than sharing a broad one, and the more summed similarity, the more likely a "yes". Because similarity is added across examples rather than requiring one concept to fit them all, a target resembling only one of the examples (e.g. 17 after 16, 8, 64) still draws a substantial share of "yes" answers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/concept_mediated_exemplar_similarity.py`

### exemplar_feature_similarity_gcm — rank 21, 306.0 ± 28.3 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1213.2

**Hypothesis:** People do not entertain explicit rules; they judge the target by its summed psychological similarity to the remembered examples (an exemplar / generalized-context model). Similarity decays exponentially with distance in a feature space where each number is described by its magnitude and by its surface properties (parity, last digit, tens digit, divisibility by 3/4/5, being prime or a square), with attention split between magnitude and properties; the more similar the target is to the examples on average, the more likely a "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`

### digit_feature_exemplar_similarity — rank 22, 324.8 ± 30.2 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1232.1

**Hypothesis:** People judge a target by its featural resemblance to the examples in decimal notation: each example's similarity to the target falls off exponentially with whether the two differ in their ones digit and whether they differ in their tens digit, and the probability of saying yes grows with the target's summed (averaged) similarity to the shown examples, as in an exemplar model over digit features rather than a search over explicit rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/digit_feature_exemplar_similarity.py`

### tenenbaum_rules_similarity — rank 23, 341.8 ± 41.7 nats behind the best (8.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1249.1

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/tenenbaum_rules_similarity.py`

### spread_adaptive_magnitude_gradient — rank 24, 378.9 ± 31.2 nats behind the best (12.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1286.1

**Hypothesis:** People judge a target by its numerical closeness to the nearest shown example, with an exponentially decaying (Shepard) generalization gradient whose width is not fixed but scales with how spread out the examples are: tightly clustered examples (e.g. 60, 62, 64) produce a sharp gradient that says "yes" only to nearby numbers, while widely scattered examples produce a broad gradient that extends "yes" to distant numbers. The single claim is this spread-adaptive magnitude similarity, with no rules or digit features.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment1/model_loop/models/spread_adaptive_magnitude_gradient.py`

## Pruned models (out of the set; narrowest margin first)

No model has been pruned yet in this project.
