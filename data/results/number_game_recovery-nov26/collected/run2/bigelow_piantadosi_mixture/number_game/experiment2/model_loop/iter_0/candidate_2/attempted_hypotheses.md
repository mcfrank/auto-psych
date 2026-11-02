# Tried before

27 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_feature_similarity_gcm — pruned (experiment1 end of experiment)

**Outcome:** 306.2 nats behind without_replacement_free_outlier_rate (11.0× dse)

**Hypothesis:** People do not entertain explicit rules; they judge the target by its summed psychological similarity to the remembered examples (an exemplar / generalized-context model). Similarity decays exponentially with distance in a feature space where each number is described by its magnitude and by its surface properties (parity, last digit, tens digit, divisibility by 3/4/5, being prime or a square), with attention split between magnitude and properties; the more similar the target is to the examples on average, the more likely a "yes".

### weak_sampling_concept_generalization — pruned (experiment1 end of experiment)

**Outcome:** 211.2 nats behind without_replacement_free_outlier_rate (10.0× dse)

**Hypothesis:** People entertain the same mathematical rules and numeric ranges as candidate concepts, but treat the examples only as labelled members of the concept rather than as random draws from it (weak sampling, no size principle): every concept consistent with the examples stays credible in proportion to its prior, however broad it is, so adding examples rules concepts out but never sharpens generalization toward the narrowest one. The sharpest disagreement with the best (strong-sampling) model is on multi-example sets that fit both a narrow rule and broader ones (e.g. 16,8,64 or 83,38,58,68): this model predicts broad, graded "yes" to targets in wide consistent ranges and broad rules, where the size-principle model predicts confident yes only for the narrow rule.

### digit_feature_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 325.1 nats behind without_replacement_free_outlier_rate (10.9× dse)

**Hypothesis:** People judge a target by its featural resemblance to the examples in decimal notation: each example's similarity to the target falls off exponentially with whether the two differ in their ones digit and whether they differ in their tens digit, and the probability of saying yes grows with the target's summed (averaged) similarity to the shown examples, as in an exemplar model over digit features rather than a search over explicit rules.

### concept_learning_with_guess_lapse — pruned (experiment1 end of experiment)

**Outcome:** 18.4 nats behind without_replacement_free_outlier_rate (4.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules with separately weighted priors plus every interval of 1-100 (size principle, 10% of examples random), but on a fixed fraction of trials they do not consult the inferred concept at all and instead answer by guessing, saying "yes" at their own baseline rate. The one change is this response-lapse component, which lets the model explain "no" answers to targets the concept clearly contains (even the shown examples) and stray "yes" answers to targets it clearly excludes.

### tenenbaum_digit_concepts — pruned (experiment1 end of experiment)

**Outcome:** 11.3 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 18

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over Tenenbaum's mathematical rules and size-weighted intervals, but they also read the examples as written numerals, so their hypothesis space additionally contains digit concepts — "contains the digit d" (for each digit 0–9) and "ends in 0" — sharing the rules' equal prior. The one change is adding these digit concepts, which predicts that examples sharing a digit (e.g. 83, 38, 58, 68) lead people to say yes to other numbers containing that digit (e.g. 84, 87), even when those numbers are far from any arithmetic rule.

### order_weighted_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 13.8 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 18

**Hypothesis:** People do Bayesian concept learning over mathematical rules, digit concepts and numeric intervals (the size principle), but they read the examples one after another and do not weigh them equally: their attention to an example shifts systematically with its position in the sequence, so later-read examples (recency) or earlier-read examples (primacy) count for more of the evidence about the concept. The one claim is this order-dependent weighting of each example's evidence — the total amount of evidence stays the same — which predicts that the same set shown in a different order generalizes more toward the concepts favoured by the emphasized examples.

### conservative_evidence_digit_concepts — pruned (experiment1 end of experiment)

**Outcome:** 15.7 nats behind without_replacement_free_outlier_rate (2.1× dse)

**Hypothesis:** People do Bayesian concept learning over Tenenbaum's mathematical rules, digit concepts and size-weighted intervals, but they are conservative updaters: they underweight (or overweight) the evidence each example provides, so the likelihood of the examples under each concept enters their belief only to a fitted power rather than at full strength. This single mis-weighting predicts that generalization tightens toward the narrowest consistent concept more slowly (or faster) with added examples than an ideal size-principle learner would, so multi-example sets still yield graded "yes" to broader consistent concepts.

### smallest_rule_commitment — pruned (experiment1 end of experiment)

**Outcome:** 197.3 nats behind without_replacement_free_outlier_rate (9.9× dse)

**Hypothesis:** People do not weigh many candidate concepts at once; they search for the single most specific rule that covers every example (the smallest of the arithmetic and digit rules such as "powers of 2", "multiples of 7", "ends in 3" or "contains an 8"), commit to it, and answer yes to a target if and only if it obeys that rule, falling back to the range spanned by the examples only when no rule fits them all. Apart from response noise (a fixed rate of yes inside the chosen rule and of yes outside it), generalization is all-or-none with no graded similarity or averaging over alternative concepts.

### digit_concepts_inferred_outlier_rate — pruned (experiment1 end of experiment)

**Outcome:** 12.6 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 18

**Hypothesis:** Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning with the size principle over Tenenbaum's arithmetic rules, size-weighted intervals and the digit concepts ("contains the digit d", "ends in 0"), while allowing that any example may be a stray random number from 1–100 — and how often they believe the program produces such strays is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate, which predicts that people discount an example that fails to fit a tight concept as a stray rather than abandoning the concept for a broader one, so narrow rules survive a misfitting example and generalization to their members stays strong.

### digit_concepts_with_guess_lapse — pruned (experiment1 end of experiment)

**Outcome:** 8.3 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 18

**Hypothesis:** Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning over Tenenbaum's mathematical rules, the digit concepts ("contains the digit d", "ends in 0") and size-weighted intervals exactly as in that model, but on a small fixed fraction of trials they do not consult the inferred concept and instead guess, saying "yes" at their own baseline rate. The one change is this response-lapse component (taken from `concept_learning_with_guess_lapse`), which predicts occasional "no" answers to targets the concept surely contains and occasional "yes" answers to targets it surely excludes, instead of near-certain responses there.

### equal_prior_concepts_with_guess_lapse — pruned (experiment1 end of experiment)

**Outcome:** 1.3 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 18

**Hypothesis:** Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle and 10% of examples random, and on a fixed fraction of trials they ignore the concept and guess "yes" at their own baseline rate — but they hold no learned preference among the rules: every rule gets the same prior weight. The one change is replacing the 45 separately inferred rule weights with a single shared rule prior, which claims that the lapse model's advantage comes from occasional guessing, not from idiosyncratic rule preferences, so the model should generalize as well with 43 fewer free parameters.

### first_example_anchored_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 30.7 nats behind without_replacement_free_outlier_rate (2.9× dse)

**Hypothesis:** People do Bayesian concept learning over the same arithmetic rules, digit concepts and numeric intervals as the best model (size principle, equal rule prior, occasional guessing), but they build their guess about the concept from the first example they read and treat it as a sure member of the concept: only the examples that follow can be dismissed as stray random numbers. This primacy-of-the-anchor claim predicts that a set whose first example misfits a tight rule shared by the later ones generalizes broadly (the rule must include the anchor), whereas the same misfit shown last is discounted and the tight rule survives.

### spread_adaptive_magnitude_gradient — pruned (experiment1 end of experiment)

**Outcome:** 379.2 nats behind without_replacement_free_outlier_rate (12.3× dse)

**Hypothesis:** People judge a target by its numerical closeness to the nearest shown example, with an exponentially decaying (Shepard) generalization gradient whose width is not fixed but scales with how spread out the examples are: tightly clustered examples (e.g. 60, 62, 64) produce a sharp gradient that says "yes" only to nearby numbers, while widely scattered examples produce a broad gradient that extends "yes" to distant numbers. The single claim is this spread-adaptive magnitude similarity, with no rules or digit features.

### soft_confidence_criterion_decision — pruned (experiment1 end of experiment)

**Outcome:** 7.6 nats behind without_replacement_free_outlier_rate (2.2× dse)

**Hypothesis:** People do Bayesian concept learning exactly as in the simplest digit-concept model (equal-prior mathematical and digit rules plus size-weighted intervals), but they do not answer "yes" with the probability that the concept contains the target; instead they compare their degree of belief against a personal criterion of how sure they must be, answering "yes" when their confidence that the target belongs exceeds that criterion, with graded (noisy) sensitivity around it. This single decision-threshold rule predicts that moderately probable targets are pushed toward "no" (or "yes") depending on where the criterion sits, and that near-certain beliefs are softened or sharpened by how steeply responses track the confidence-versus-criterion gap.

### iter2_candidate3 — pruned (experiment1 end of experiment)

**Outcome:** 4.0 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 18

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle, 10% of examples random, and a guess lapse at their own baseline "yes" rate — but the rules are not all equally salient: each rule's prior weight is a draw from a shared population of rule saliences whose spread is inferred, so a rule departs from the common weight only as far as the answers demand. The one change is replacing the single shared rule prior with this partially pooled (hierarchical) rule prior, which claims that some rules (e.g. "odd numbers") are systematically less or more attractive than others, yet predicts generalization closer to the equal-prior model than 45 freely inferred weights would.

### specificity_biased_rule_prior_lapse — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'rule_breadth_prior_concepts' (p_yes RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_breadth_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100. They apply the size principle, treat 10% of examples as random, and on a fixed fraction of trials ignore the concept and guess "yes" at their own baseline rate. The one change is that their prior over rules is neither 45 free weights nor flat. It follows a specificity bias: a rule's prior plausibility falls (or rises) as a power of how many numbers it contains, so narrow rules like "powers of 3" are favoured over broad ones like "odd numbers" before any example is seen. This predicts weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) than an equal-prior learner, with the strength of the bias set by a single parameter.

### occam_specificity_prior_all_concepts — pruned (experiment1 end of experiment)

**Outcome:** 2.3 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 18

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random, occasional guessing at their own yes rate), but they bring one general Occam-style specificity bias to every candidate concept, rules and numeric ranges alike: before seeing any examples, a concept's prior plausibility falls (or rises) as a power of how many numbers it contains. The one change is replacing the flat priors within the rules and within the intervals with this single shared size-graded prior, which predicts not only weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) but also tighter, more local "yes" ranges around clustered examples than an equal-prior learner, with both effects set by the same one parameter.

### concept_mediated_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 269.3 nats behind without_replacement_free_outlier_rate (10.8× dse)

**Hypothesis:** People do not infer a single concept that covers all the examples; they store each example and judge the target by its summed (averaged) similarity to those stored examples, as in an exemplar model. The similarity of the target to one example is how likely the two are to share a number concept — an arithmetic or digit rule (e.g. "powers of 2", "contains a 3") or a numeric range — judged from that example alone, so sharing a narrow property counts for more than sharing a broad one, and the more summed similarity, the more likely a "yes". Because similarity is added across examples rather than requiring one concept to fit them all, a target resembling only one of the examples (e.g. 17 after 16, 8, 64) still draws a substantial share of "yes" answers.

### pedagogical_sampling_concepts — pruned (experiment1 end of experiment)

**Outcome:** 2.0 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 18

**Hypothesis:** People treat the examples as chosen on purpose by a helpful teacher rather than drawn at random from the hidden concept: they assume the program picks, from among the concept's members, numbers that would best lead a learner to that concept (numbers that few other plausible concepts contain), and they infer the concept by reasoning about why these particular examples were chosen. Over the same arithmetic rules, digit concepts and numeric ranges (with occasional guessing), this pedagogical reading predicts that an example typical of many concepts (e.g. a small even number) says less about the concept than a distinctive one, so generalization follows the concept the examples single out more than the size principle alone implies, with the strength of this teacher assumption inferred from the answers.

### magnitude_biased_example_sampling — rejected (experiment1 round 4 candidate 0 lens 1)

**Outcome:** predicts like existing model 'concepts_sampled_without_replacement' (p_yes RMSE 0.00104 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of concepts_sampled_without_replacement, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules and numeric ranges as the best model (size-graded rule prior, distinct examples drawn without replacement, 10% strays, occasional guessing), but they believe the program does not pick a concept's members equally often: it favours numbers of a particular magnitude (small numbers more often than large, or the reverse), so an example's evidence for a concept depends on how much of the concept's magnitude-weighted mass it carries, not just on the concept's member count. This single magnitude-biased sampling assumption disagrees most sharply with the best model on sets made of large numbers (e.g. 92, 82, 27, 42 or 83, 38, 58, 68) versus small ones (e.g. 16, 8 or 26, 2): with a small-number bias, large examples are unexpectedly strong evidence for concepts whose members are mostly large (high ranges, "contains 8", "ends in 2"), and small examples are weak evidence for small-number rules such as powers of 2.

### description_length_rule_cost — rejected (experiment1 round 4 candidate 2 lens 3)

**Outcome:** predicts like existing model 'without_replacement_size_principle' (p_yes RMSE 0.00137 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of without_replacement_size_principle, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules, digit concepts and numeric ranges. They assume the program shows distinct numbers, sampled from the concept, and they sometimes guess. Considering a rule costs mental effort that grows with how long the rule takes to state, so wordier rules are less often entertained. Single-word rules like "odd", "squares" or "primes" are cheapest. Rules with a parameter ("multiples of 7", "ends in 3", "contains a 5") cost more. Rules with an exception ("powers of 2 except 32") cost the most. This description-length cost replaces any preference for broad or narrow rules. It predicts fewer "yes" answers for exception rules and parameterised families than their size alone implies, while simple rules keep their pull.

### without_replacement_inferred_stray_rate — rejected (experiment1 round 4 candidate 4 refine incumbent concepts_sampled_without_replacement)

**Outcome:** predicts like existing model 'without_replacement_free_outlier_rate' (p_yes RMSE 0.00047 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of without_replacement_free_outlier_rate, not a new hypothesis.

**Hypothesis:** Refinement of `concepts_sampled_without_replacement` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with the size-graded rule prior, occasional guessing at their own yes rate, and strong sampling without replacement (each further example a draw from the concept's members not yet shown). The one change, a component taken from `breadth_prior_concepts_free_outlier_rate`, is that how often people believe the program emits a stray random example is inferred from their answers rather than fixed at one in ten. The claim is that a without-replacement learner also discounts a misfitting example as a stray, at a rate of its own, so tight rules and short ranges survive an odd-one-out example (e.g. 85, 19, 91 or 14, 47) instead of giving way to broad concepts.

### occam_prior_without_replacement — pruned (experiment1 end of experiment)

**Outcome:** 1.4 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 18

**Hypothesis:** Refinement of `occam_specificity_prior_all_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with one shared Occam-style specificity bias that makes every concept's prior (rules and ranges alike) fall or rise as a power of its size, 10% of examples random and occasional guessing at their own yes rate. The one change, taken from `without_replacement_size_principle`, is that people take the program's examples to be distinct, so each further example is a draw from the concept's members not yet shown (strong sampling without replacement). Together with the shared specificity prior, this predicts that examples that nearly use up a small rule or a short range (e.g. 16, 8, 64 for powers of 2, or 60, 62, 64 for a tight range) make that concept even more credible, so people say "yes" less to members of broad rules and wide ranges than either change alone predicts.

### round_number_bounded_ranges — pruned (experiment1 end of experiment)

**Outcome:** 6.0 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 18

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules as the best model (size-graded rule prior, distinct examples drawn without replacement, 10% strays, occasional guessing), but they think of numeric ranges in round-number terms: the ranges they entertain begin and end at multiples of five ("from 60 to 70", "from 35 to 85"), not at arbitrary numbers. This disagrees most sharply with the best model at the edges of the examples' span: after clustered examples such as 60, 62, 64 or 92, 82, 27, 42, "yes" should stay high right up to the next multiple of five beyond the outermost example and then drop in a step, rather than fall off smoothly from the examples themselves.

### win_stay_lose_shift_concept_testing — pruned (experiment1 end of experiment)

**Outcome:** 68.7 nats behind without_replacement_free_outlier_rate (6.1× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they read the examples one at a time while holding a single candidate concept (from the same arithmetic and digit rules, with a breadth-graded rule prior, and numeric ranges), keeping it as long as each new example is one of its members ("win-stay") and, only when an example falls outside it, discarding it and drawing a fresh concept in proportion to how well concepts explain all the examples so far ("lose-shift"). Because a concept that survives is never re-weighed, later examples that fit a broad concept do not sharpen belief toward a narrower one, so this sequential hypothesis-testing process predicts more "yes" answers to members of broad rules and wide ranges after several consistent examples (e.g. 16, 8, 64 or 60, 62, 64) than full Bayesian updating, and it makes the answer depend on which example came first.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 342.1 nats behind without_replacement_free_outlier_rate (8.2× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 19.6 nats behind without_replacement_free_outlier_rate (3.6× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.
