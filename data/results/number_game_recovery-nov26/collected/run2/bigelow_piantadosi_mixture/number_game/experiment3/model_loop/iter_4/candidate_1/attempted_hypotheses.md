# Tried before

68 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### rule_breadth_prior_concepts — pruned (experiment2 end of experiment)

**Outcome:** 6.1 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 24 of 26

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random), with occasional guessing at their own yes rate — but before seeing any examples they already prefer rules by their breadth, giving each rule a prior weight that rises or falls as a power of how many numbers it contains (a preference for specific, "interesting" rules over broad ones, or the reverse). The one change is replacing the flat prior across rules with this single size-graded rule prior, which predicts that broad rules such as "odd" or "even" win less (or more) of the posterior than an equal-prior learner gives them, e.g. fewer "yes" answers to arbitrary odd numbers after 85, 19, 91.

### specificity_biased_rule_prior_lapse — rejected (experiment1 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'rule_breadth_prior_concepts' (p_yes RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_breadth_prior_concepts, not a new hypothesis.

**Hypothesis:** Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100. They apply the size principle, treat 10% of examples as random, and on a fixed fraction of trials ignore the concept and guess "yes" at their own baseline rate. The one change is that their prior over rules is neither 45 free weights nor flat. It follows a specificity bias: a rule's prior plausibility falls (or rises) as a power of how many numbers it contains, so narrow rules like "powers of 3" are favoured over broad ones like "odd numbers" before any example is seen. This predicts weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) than an equal-prior learner, with the strength of the bias set by a single parameter.

### occam_specificity_prior_all_concepts — pruned (experiment1 end of experiment)

**Outcome:** 2.3 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 18

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random, occasional guessing at their own yes rate), but they bring one general Occam-style specificity bias to every candidate concept, rules and numeric ranges alike: before seeing any examples, a concept's prior plausibility falls (or rises) as a power of how many numbers it contains. The one change is replacing the flat priors within the rules and within the intervals with this single shared size-graded prior, which predicts not only weaker generalization to broad rules (e.g. all odd numbers after 85, 19, 91) but also tighter, more local "yes" ranges around clustered examples than an equal-prior learner, with both effects set by the same one parameter.

### counting_by_twos_run_concepts — pruned (experiment2 end of experiment)

**Outcome:** 7.4 nats behind margin_and_heading_range_prior (2.1× dse)

**Hypothesis:** Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same arithmetic and digit rules (with their size-graded rule prior) and every interval of 1-100, with the size principle, 10% of examples random and occasional guessing. They also entertain one more kind of concept, "counting by twos" runs: a bounded stretch of every other number, such as 60, 62, 64, 66 or 31, 33, 35, i.e. the even or the odd numbers within a limited range. All such runs are equally likely a priori, and the family as a whole takes an inferred share of prior belief relative to the rules. This predicts that examples spaced by two (e.g. 60, 62, 64) draw "yes" to nearby same-parity numbers but not to same-parity numbers far away, a pattern that neither "even numbers" nor a numeric range gives.

### concept_mediated_exemplar_similarity — pruned (experiment1 end of experiment)

**Outcome:** 269.3 nats behind without_replacement_free_outlier_rate (10.8× dse)

**Hypothesis:** People do not infer a single concept that covers all the examples; they store each example and judge the target by its summed (averaged) similarity to those stored examples, as in an exemplar model. The similarity of the target to one example is how likely the two are to share a number concept — an arithmetic or digit rule (e.g. "powers of 2", "contains a 3") or a numeric range — judged from that example alone, so sharing a narrow property counts for more than sharing a broad one, and the more summed similarity, the more likely a "yes". Because similarity is added across examples rather than requiring one concept to fit them all, a target resembling only one of the examples (e.g. 17 after 16, 8, 64) still draws a substantial share of "yes" answers.

### pedagogical_sampling_concepts — pruned (experiment1 end of experiment)

**Outcome:** 2.0 nats behind without_replacement_free_outlier_rate; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 18

**Hypothesis:** People treat the examples as chosen on purpose by a helpful teacher rather than drawn at random from the hidden concept: they assume the program picks, from among the concept's members, numbers that would best lead a learner to that concept (numbers that few other plausible concepts contain), and they infer the concept by reasoning about why these particular examples were chosen. Over the same arithmetic rules, digit concepts and numeric ranges (with occasional guessing), this pedagogical reading predicts that an example typical of many concepts (e.g. a small even number) says less about the concept than a distinctive one, so generalization follows the concept the examples single out more than the size principle alone implies, with the strength of this teacher assumption inferred from the answers.

### breadth_prior_concepts_free_outlier_rate — pruned (experiment2 end of experiment)

**Outcome:** 6.7 nats behind margin_and_heading_range_prior (2.2× dse)

**Hypothesis:** Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules (with the same breadth-graded rule prior) and every interval of 1-100, with the size principle and occasional guessing at their own yes rate — but how often they believe the program emits a stray random example is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate (the component taken from `digit_concepts_inferred_outlier_rate`), which claims that part of what the lapse now absorbs is really people discounting a misfitting example as a stray, so tight rules survive an odd-one-out example (e.g. 85, 19, 91 or 14, 47) instead of being abandoned for broad ones.

### concepts_sampled_without_replacement — pruned (experiment2 end of experiment)

**Outcome:** 7.2 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 26 of 26

**Hypothesis:** Refinement of `rule_breadth_prior_concepts`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with a size-graded rule prior, 10% of examples random and occasional guessing at their own yes rate. They also take it that the program shows distinct numbers, so each further example is a draw from the concept's members not yet shown (strong sampling without replacement). The one change is this sampling assumption. It sharpens the size principle most for small concepts that are nearly used up by the examples (e.g. 8, 64 or 16, 8, 64 strongly favour "powers of 2", or 1, 8, 27 "cubes"), so people say "yes" less often to members of broader rules and wide ranges than a with-replacement learner predicts.

### without_replacement_size_principle — pruned (experiment2 end of experiment)

**Outcome:** 5.1 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 26

**Hypothesis:** Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (equal rule prior, 10% of examples random, occasional guessing at their own yes rate), but they understand that the program shows distinct examples, so each further example is read as a draw from the concept's members not yet shown (sampling without replacement) rather than from all its members again. The one change is this without-replacement size principle, which adds no free parameter and predicts that several examples sharpen generalization toward small concepts more than the standard model does: a few examples that nearly exhaust a small rule (e.g. 16, 8, 64 for powers of 2) or a short range make that concept far more credible than broader alternatives.

### magnitude_biased_example_sampling — rejected (experiment1 round 4 candidate 0 lens 1)

**Outcome:** predicts like existing model 'concepts_sampled_without_replacement' (p_yes RMSE 0.00104 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of concepts_sampled_without_replacement, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules and numeric ranges as the best model (size-graded rule prior, distinct examples drawn without replacement, 10% strays, occasional guessing), but they believe the program does not pick a concept's members equally often: it favours numbers of a particular magnitude (small numbers more often than large, or the reverse), so an example's evidence for a concept depends on how much of the concept's magnitude-weighted mass it carries, not just on the concept's member count. This single magnitude-biased sampling assumption disagrees most sharply with the best model on sets made of large numbers (e.g. 92, 82, 27, 42 or 83, 38, 58, 68) versus small ones (e.g. 16, 8 or 26, 2): with a small-number bias, large examples are unexpectedly strong evidence for concepts whose members are mostly large (high ranges, "contains 8", "ends in 2"), and small examples are weak evidence for small-number rules such as powers of 2.

### round_number_salient_sampling — pruned (experiment2 end of experiment)

**Outcome:** 9.9 nats behind margin_and_heading_range_prior (2.3× dse)

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules (with a size-graded rule prior) and every interval of 1-100, with occasional guessing, treating the examples as distinct draws from the hidden concept (strong sampling without replacement, with some stray random examples). The one new claim is about how they believe the program picks numbers: not uniformly, but with round numbers (multiples of 5 and 10) more or less likely to be produced than other numbers, with the strength and direction of this round-number bias inferred from their answers. Because a concept's members are then weighted by how readily the program would emit them, this predicts that concepts made mostly of round numbers (e.g. "multiples of 10", "ends in 0" or "ends in 5") gain or lose credibility relative to other rules and ranges. It also predicts that a round example says less about the concept than a non-round one does, or more.

### description_length_rule_cost — rejected (experiment1 round 4 candidate 2 lens 3)

**Outcome:** predicts like existing model 'without_replacement_size_principle' (p_yes RMSE 0.00137 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of without_replacement_size_principle, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over arithmetic rules, digit concepts and numeric ranges. They assume the program shows distinct numbers, sampled from the concept, and they sometimes guess. Considering a rule costs mental effort that grows with how long the rule takes to state, so wordier rules are less often entertained. Single-word rules like "odd", "squares" or "primes" are cheapest. Rules with a parameter ("multiples of 7", "ends in 3", "contains a 5") cost more. Rules with an exception ("powers of 2 except 32") cost the most. This description-length cost replaces any preference for broad or narrow rules. It predicts fewer "yes" answers for exception rules and parameterised families than their size alone implies, while simple rules keep their pull.

### without_replacement_free_outlier_rate — pruned (experiment2 end of experiment)

**Outcome:** 7.1 nats behind margin_and_heading_range_prior (2.3× dse)

**Hypothesis:** Refinement of `concepts_sampled_without_replacement`: people do Bayesian concept learning over the same 45 arithmetic and digit rules (with the size-graded rule prior) and every interval of 1-100, reading each further example as a draw from the concept's members not yet shown (strong sampling without replacement), with occasional guessing at their own yes rate. The one change (the component taken from `breadth_prior_concepts_free_outlier_rate`) is that how often they believe the program shows a stray random number is a property of the learner, inferred from their answers, rather than fixed at one in ten — so a misfitting example (e.g. 14, 47 or 85, 19, 91) is discounted as a stray at the learner's own rate, letting tight, nearly exhausted concepts survive it instead of yielding to broad rules and wide ranges.

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

### shown_examples_not_repeated — pruned (experiment2 end of experiment)

**Outcome:** 8.4 nats behind margin_and_heading_range_prior (2.3× dse)

**Hypothesis:** Refinement of `concepts_sampled_without_replacement` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, with the size-graded rule prior, 10% of examples random, occasional guessing at their own yes rate, and the belief that the program shows distinct numbers, so each further example is a draw from the concept's members not yet shown. The one change follows from that same belief: when the number asked about is one of the examples already shown, people sometimes (at a rate inferred from their answers) read "would the program also produce it?" as asking whether it would come up again as a new example, and since the program does not repeat itself they answer "no". This predicts fewer "yes" answers to targets that are themselves examples than the concept's membership implies (about two-thirds rather than three-quarters), while leaving every other judgment governed by the incumbent's concept learning and making the guessing lapse smaller.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 342.1 nats behind without_replacement_free_outlier_rate (8.2× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 19.6 nats behind without_replacement_free_outlier_rate (3.6× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### narrow_range_specificity_bias — pruned (experiment2 end of experiment)

**Outcome:** 4.8 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 26

**Hypothesis:** People infer the hidden concept by Bayesian inference over arithmetic and digit rules and numeric ranges. They read the examples as distinct numbers drawn from the concept, allow that some are stray, and sometimes guess. The one distortion is a specificity bias in how they think about ranges: before seeing any examples, they already judge a narrow stretch of numbers a likelier program than a wide one, so a range's prior plausibility falls (or rises) as a power of its width. The rules themselves keep an equal prior. This predicts tighter "yes" gradients around clustered examples (e.g. 60, 62, 64) and fewer "yes" answers far beyond the examples' span than an unbiased learner gives, while generalization by rules (e.g. powers of 2 after 16, 8, 64) is left unchanged.

### pure_rule_induction_no_ranges — pruned (experiment2 end of experiment)

**Outcome:** 456.3 nats behind margin_and_heading_range_prior (14.5× dse)

**Hypothesis:** People treat the task purely as rule induction: they weigh only the arithmetic and digit rules (odd, even, primes, squares, powers, multiples, "ends in d", "contains d") by how well each explains the distinct examples shown (the size principle, sampling without replacement), and when no rule fits they fall back on "the program produces any number at all", with occasional guessing. They never entertain numeric ranges, so this predicts no graded magnitude generalization: a target near the examples (e.g. 61 after 60, 62, 64) is no more likely to get a "yes" than a distant number that fits the same rules.

### position_graded_example_trust — pruned (experiment2 end of experiment)

**Outcome:** 5.4 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 21 of 26

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules and numeric ranges as the best model (equal rule prior, distinct examples drawn without replacement, occasional guessing), but they read the examples in order and their trust in each one shifts steadily along the sequence: the chance they put on an example being a stray random number rises or falls by a fixed step with each later position (recency or primacy of trust), with the average stray rate unchanged. This predicts that the same misfitting example is discounted more readily when it comes at the distrusted end of the sequence (e.g. 47 after 14 versus 14 after 47), so tight rules survive an odd-one-out example in one position but give way to broad concepts in the other.

### narrow_range_prior_without_replacement — rejected (experiment2 round 0 candidate 4 refine incumbent without_replacement_size_principle)

**Outcome:** predicts like existing model 'narrow_range_specificity_bias' (p_yes RMSE 0.00140 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of narrow_range_specificity_bias, not a new hypothesis.

**Hypothesis:** Refinement of `without_replacement_size_principle`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100. As in that model, they read the examples as distinct draws (sampling without replacement), with 10% random examples. The one change is that, before seeing any examples, they find narrow numeric ranges more plausible than wide ones: each range's prior weight falls as a power of its width, with the strength of this preference inferred from the answers. This predicts tighter "yes" regions around clustered examples, and fewer "yes" answers to distant numbers that only a wide range would cover. Their occasional off-concept answers are a bare "yes", which is what the incumbent's guessing rate converges to on this data, so the model has no more free parameters than the incumbent.

### occam_range_width_prior — rejected (experiment2 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'narrow_range_specificity_bias' (p_yes RMSE 0.00044 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of narrow_range_specificity_bias, not a new hypothesis.

**Hypothesis:** Refinement of `occam_prior_without_replacement`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, reading each example as a distinct draw from the concept's members not yet shown (sampling without replacement), with 10% stray examples and occasional guessing. The one change is where the Occam bias applies. It works only on numeric ranges: before seeing examples, a range's plausibility falls (or rises) as a power of its width, and all rules keep an equal prior. People's preference for narrow concepts is then a magnitude habit about short stretches of the number line, not a preference among arithmetic rules. This predicts that after a single example (e.g. 10 or 49), "yes" falls off faster with distance beyond the immediate neighbours than an equal-prior learner predicts, while rule-based generalization (e.g. powers of 2 after 16, 8, 64) is unchanged.

### final_digit_rule_discount — pruned (experiment2 end of experiment)

**Outcome:** 7.9 nats behind margin_and_heading_range_prior (2.1× dse)

**Hypothesis:** Refinement of `without_replacement_size_principle` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100, reading each further example as a draw from the concept's members not yet shown (sampling without replacement), with 10% of examples random and occasional guessing at their own yes rate. The one change is that the final-digit rules ("ends in 3", "ends in 7", ...) are not as plausible a program as the other rules: before seeing examples, people hold the "ends in d" rules at a different prior weight (inferred from their answers; lower if a shared last digit reads as a coincidence of notation rather than a rule), while all other rules keep an equal prior. This predicts fewer (or more) "yes" answers to numbers that share only the examples' final digit (e.g. 53 after 13, 73) than the incumbent gives, while range-based and arithmetic generalization are left as they are.

### relative_width_range_prior — pruned (experiment2 end of experiment)

**Outcome:** 5.7 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 22 of 26

**Hypothesis:** Refinement of `without_replacement_size_principle` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules and every numeric range of 1-100, reading the examples as distinct draws (sampling without replacement), with 10% stray examples and occasional guessing at their own yes rate. The one change is a Weber-like sense of how narrow a range is: before seeing examples, people judge a range more (or less) plausible as a power of its width *relative to its magnitude* (width divided by the range's midpoint), so "60 to 64" counts as a tight, specific range while "2 to 6", equally wide in absolute terms, counts as a broad one. This predicts sharper, more local "yes" gradients around clustered large examples (e.g. 60, 62, 64 or 83, 77) than around clustered small ones (e.g. 3, 7 or 5, 9), a magnitude dependence that neither the flat range prior nor an absolute-width prior produces.

### separate_rule_and_range_specificity — pruned (experiment2 end of experiment)

**Outcome:** 7.1 nats behind margin_and_heading_range_prior (2.2× dse)

**Hypothesis:** Refinement of `narrow_range_specificity_bias`: people do Bayesian concept learning over arithmetic and digit rules and every numeric range, reading the examples as distinct draws from the concept (with some strays) and sometimes guessing, and they already judge a narrow range a likelier program than a wide one, as a power of its width. The one change, the component taken from `rule_breadth_prior_concepts`, is that the rules no longer share an equal prior: a second, separate breadth bias makes a rule's prior rise or fall as a power of how many numbers it contains. The claim is that people bring two distinct specificity habits, one about stretches of the number line and one about arithmetic rules, each with its own strength and direction (the data suggest a narrow-range preference alongside a slight preference for broad rules). This predicts that broad rules such as "odd numbers" (after 85, 19, 91) or "even numbers" win a different share of "yes" than the range-only bias allows, while the tight "yes" gradients around clustered examples (e.g. 60, 62, 64) are kept.

### set_common_feature_contrast — pruned (experiment2 end of experiment)

**Outcome:** 631.6 nats behind margin_and_heading_range_prior (17.7× dse)

**Hypothesis:** People judge the target by its feature-contrast similarity to the example set as a whole: they notice the properties every example shares (parity, final digit, a contained digit, being a square, power or multiple of something), and the target counts as similar to the extent it keeps those shared properties and dissimilar to the extent it lacks them, with rarer, more distinctive shared properties (e.g. "power of 2") weighing far more than common ones (e.g. "even"); on magnitude, similarity falls off steadily with how far the target lies outside the stretch the examples span. The more similar on this metric, the more likely a "yes" — there is no weighing of alternative hypotheses, only matching against the set's common features.

### sampled_concept_majority_vote — pruned (experiment2 end of experiment)

**Outcome:** 6.2 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 25 of 26

**Hypothesis:** People infer the hidden concept exactly as the simplest current model does (Bayesian concept learning over arithmetic and digit rules and numeric ranges, distinct examples, some strays, occasional guessing), but they do not answer by consulting their whole belief: they call to mind a small number of candidate concepts, each drawn in proportion to its posterior plausibility, and answer "yes" when most of those sampled concepts contain the target (a majority vote over a few posterior samples). How many concepts they sample is inferred from their answers; sampling one is pure probability matching, while sampling more pushes moderately likely targets toward a firm "yes" or "no", so "yes" rates should be more extreme than the posterior on items the posterior calls fairly likely or fairly unlikely.

### digit_sum_concepts — pruned (experiment2 end of experiment)

**Outcome:** 3.6 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 26

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules and numeric ranges as the best model: distinct examples drawn without replacement, some strays, occasional guessing, and a prior that favours narrow ranges. They also read the numbers as written numerals and entertain one more kind of concept, "numbers whose digits add up to s" (for example 18, 27, 36, 45 or 13, 22, 31, 40). All 18 such digit-sum concepts are equally likely a priori, and the family as a whole takes an inferred share of the prior belief given to rules. This predicts that examples sharing a digit sum (e.g. 36, 45, 54, 81) lead to "yes" for other numbers with the same digit sum, such as 63, 72, 90 or 27, beyond what "multiples of 9" or a range gives, and to fewer "yes" answers for other members of those broader concepts.

### heterogeneous_range_specificity_bias — pruned (experiment2 end of experiment)

**Outcome:** 4.7 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 26

**Hypothesis:** Refinement of `narrow_range_specificity_bias` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules (equal prior) and every numeric range of 1-100, reading the examples as distinct draws with some strays and sometimes guessing, and each person's prior over ranges rises or falls as a power of the range's width. The one change is that the strength of this specificity bias differs from person to person: the population holds a spread of width biases, with some people strongly preferring narrow ranges and others preferring wide ones, and the spread is inferred from the answers, where the incumbent assumes a single shared bias. This predicts the extra disagreement between respondents on items of moderate confidence, such as targets just beyond a cluster like 60, 62, 64, where narrow-range people say "no" and wide-range people say "yes". It also predicts a population "yes" gradient that is sharp near the examples but has longer tails than any single shared bias gives, while generalization by rules is left unchanged.

### range_location_gradient_prior — pruned (experiment2 end of experiment)

**Outcome:** 5.9 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 23 of 26

**Hypothesis:** Refinement of `narrow_range_specificity_bias` (the incumbent): people do Bayesian concept learning over the same 45 arithmetic and digit rules (equal prior) and every numeric range of 1-100. As in the incumbent, they read the examples as distinct draws from the concept (sampling without replacement), allow 10% stray examples, sometimes guess, and bias the range prior by width. The one change is that the range prior also depends on where the range sits on the number line. Before seeing examples, people find a stretch that starts higher up (e.g. "70 to 80") more (or less) plausible than an equally wide one that starts low (e.g. "5 to 15"), and a range's prior plausibility rises or falls steadily with its lower bound. This predicts that generalization from a cluster of examples is not symmetric about it: "yes" answers spread further on one side of the examples than on the other (upward or downward, set by the bias), which a width-only prior cannot produce.

### open_ended_range_discount — pruned (experiment2 end of experiment)

**Outcome:** 4.3 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 26

**Hypothesis:** Refinement of `relative_width_range_prior`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every numeric range of 1-100, reading the examples as distinct draws (sampling without replacement) with 10% strays, occasional guessing, and the same Weber-like prior that favours ranges narrow relative to their magnitude. The one change is that a range running to an end of the number line ("anything up to 30", "anything from 70 on") is a threshold rule rather than a bounded stretch, and people hold these open-ended ranges at a different prior weight from bounded ones (inferred from their answers; the data suggest lower). This predicts fewer "yes" answers to numbers far below small examples, or far above large ones, all the way to 1 or 100, while generalization to nearby numbers on both sides of the examples' span is kept.

### multiplicative_exemplar_resemblance — pruned (experiment2 end of experiment)

**Outcome:** 1008.6 nats behind margin_and_heading_range_prior (23.1× dse)

**Hypothesis:** People do not infer one concept for the whole set; they store each example and compare the target to every stored example in turn, judging each pairwise resemblance as how likely the two numbers are to share an arithmetic or digit rule (e.g. "powers of 2", "contains a 3") or a numeric range, so sharing a narrow property counts for more than sharing a broad one. These resemblances combine multiplicatively, as a geometric mean over the examples (a conjunctive exemplar rule: the target must resemble every example, and one poor match drags the whole judgment down), and the greater this overall resemblance, the more likely a "yes". A target that matches only some of the examples (e.g. 17 after 16, 8, 64) is therefore strongly rejected, unlike in an additive exemplar model.

### sequence_direction_range_extrapolation — pruned (experiment2 end of experiment)

**Outcome:** 4.5 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 26

**Hypothesis:** People do Bayesian concept learning over the same arithmetic and digit rules, digit-sum concepts and narrow-favouring numeric ranges as the best model, but they read the order of the examples as a direction of travel (rising if the last example is above the first, falling if below) and take it as evidence that the program's range runs on in that direction, crediting ranges by how far they extend ahead of the sequence rather than behind it (with an inferred strength). The disagreement with the best model lives just outside the examples' span: this predicts more "yes" to numbers beyond the last-shown end of a set than beyond the first-shown end (e.g. 70 after 52, 60, 66 but less so after 66, 60, 52), whereas the best model ignores order for ranges and generalizes symmetrically.

### open_ended_digit_sum_concepts — rejected (experiment2 round 2 candidate 4 refine incumbent digit_sum_concepts)

**Outcome:** predicts like existing model 'digit_sum_open_ended_range_discount' (p_yes RMSE 0.00031 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of digit_sum_open_ended_range_discount, not a new hypothesis.

**Hypothesis:** Refinement of `digit_sum_concepts` (the incumbent): people do Bayesian concept learning over the same arithmetic and digit rules, the 18 "digits add up to s" concepts and every numeric range of 1-100, reading the examples as distinct draws with some strays, sometimes guessing, and with the same width-graded range prior. The one change, the component taken from `open_ended_range_discount`, is that a range running to an end of the number line ("anything up to 30", "anything from 70 on") is a threshold rule, not a bounded stretch, and people hold these open-ended ranges at a different prior weight from bounded ranges of the same width. That weight is inferred from their answers, and the data suggest it is lower. This predicts fewer "yes" answers to numbers far below small examples, or far above large ones, all the way down to 1 or up to 100, while digit-sum and rule generalization and "yes" to nearby numbers on both sides of the examples are kept.

### open_ended_ranges_with_digit_sums — pruned (experiment2 end of experiment)

**Outcome:** 3.3 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 26

**Hypothesis:** Refinement of `open_ended_range_discount`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every numeric range of 1-100. They read the examples as distinct draws with 10% strays and sometimes guess. They keep the Weber-like prior that favours ranges narrow relative to their magnitude, and they hold open-ended threshold ranges ("up to 30", "from 70 on") at a lower prior than bounded ones. The one change, the component taken from `digit_sum_concepts`, is that they also read the numbers as written numerals and entertain "numbers whose digits add up to s" as candidate concepts, all 18 equally likely, with the family taking an inferred share of the prior given to rules. This predicts that examples sharing a digit sum (e.g. 36, 45, 54, 81) draw "yes" to other numbers with that digit sum (63, 72, 27), together with fewer "yes" answers far out toward 1 or 100 beyond small or large examples than a bounded-range-only learner gives; the digit-sum concepts and the open-ended discount act on separate parts of the hypothesis space.

### digit_sum_range_location_prior — pruned (experiment2 end of experiment)

**Outcome:** 4.4 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 26

**Hypothesis:** Refinement of `digit_sum_concepts` (the incumbent): people do Bayesian concept learning over the same arithmetic and digit rules, the 18 "digits add up to s" concepts and every numeric range of 1-100, reading the examples as distinct draws with some strays, sometimes guessing, and favouring narrow ranges. The one change, the component taken from `range_location_gradient_prior`, is that the range prior also depends on where a range sits on the number line: before seeing examples, people find a stretch that starts higher up (e.g. "70 to 80") more (or less) plausible than an equally wide one that starts low (e.g. "5 to 15"), with prior plausibility rising or falling steadily with the range's lower bound (strength and direction inferred). This predicts that "yes" answers spread further on one side of a cluster of examples than on the other, while digit-sum and rule generalization are kept as in the incumbent.

### spread_coverage_expectation — pruned (experiment2 end of experiment)

**Outcome:** 3.1 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 26

**Hypothesis:** People expect the program's examples to be spread out across the whole concept rather than bunched in one corner of it, so when they weigh candidate concepts (the same arithmetic, digit and digit-sum rules and numeric ranges as the best model, with distinct examples, some strays, occasional guessing, a narrow-range preference and a discount on open-ended ranges) they credit a concept more when the examples reach across most of its members and less when the examples cover only a small stretch of it, with the strength of this coverage expectation inferred. This predicts that several examples spanning a wide stretch (e.g. 10, 50, 100) favour concepts that end near the outermost examples, so "yes" falls off faster just beyond them, and that examples bunched within a small part of a large rule (e.g. 2, 4 for "even numbers") count against that rule more than the size principle alone implies; a single example is unaffected.

### limited_concept_consideration_set — pruned (experiment2 end of experiment)

**Outcome:** 2.1 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 26

**Hypothesis:** People do not weigh every conceivable concept: they bring to mind a limited sample of candidate concepts (arithmetic and digit rules, digit-sum concepts, numeric ranges), each coming to mind in proportion to how plausible it seems before any examples, and then judge the target only by those candidates that came to mind, weighted by how well they explain the examples, as in the best model. Because a simple, a priori plausible rule is almost always among the candidates however plausible it is, the advantage of plausible concepts saturates, so relative to full Bayesian weighing this predicts somewhat more "yes" by low-prior concepts such as a particular narrow range or a digit-sum family and less by broad rules, with the size of the consideration set inferred from the answers.

### heading_extrapolation_bounded_ranges — rejected (experiment2 round 3 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'open_ended_ranges_heading_direction' (p_yes RMSE 0.00054 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of open_ended_ranges_heading_direction, not a new hypothesis.

**Hypothesis:** Refinement of `heading_direction_range_extrapolation`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays, sometimes guess, favour narrow ranges, and let the direction of the last step (up or down) stretch the ranges they consider further in the direction the sequence is heading. The one change, the component taken from `open_ended_range_discount`, is that a range running all the way to an end of the number line ("anything up to 30", "anything from 70 on") counts as a threshold rule rather than a bounded stretch, and gets its own prior weight relative to bounded ranges (inferred; the data suggest lower). The prediction: the heading still pushes "yes" answers further on the side the sequence was moving toward, but that spread stops short of 1 or 100 instead of running to the end of the line, so far-out numbers get fewer "yes" answers on both sides than the heading model gives.

### heading_extrapolation_location_prior — pruned (experiment2 end of experiment)

**Outcome:** 2.2 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 26

**Hypothesis:** Refinement of `heading_direction_range_extrapolation`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays, sometimes guess, weigh ranges by their width, and let the direction of the last step (up or down from the example before) stretch the ranges they consider further in the direction the sequence is heading. The one change, the component taken from `range_location_gradient_prior`, is that before seeing any examples people also find a range more (or less) plausible according to where it starts on the number line, with prior plausibility rising or falling steadily with its lower bound (inferred; the data suggest ranges higher up are favoured). This predicts that, on top of the push toward where the sequence is heading, "yes" spreads further above a cluster of examples than below it even for a single example or a set whose last step points down, an order-independent asymmetry that the heading model cannot produce.

### uncertain_strong_weak_sampling — rejected (experiment2 round 4 candidate 0 lens 5)

**Outcome:** predicts like existing model 'margin_beyond_examples_range_prior' (p_yes RMSE 0.00165 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of margin_beyond_examples_range_prior, not a new hypothesis.

**Hypothesis:** People are unsure how the program's examples were chosen: they partly believe the examples were sampled from the hidden concept (so a small concept that fits them all is a suspicious coincidence, the size principle) and partly believe the examples are just arbitrary numbers that happen to carry the label (so every concept that contains them all stays equally credible however broad it is), with the strength of that belief inferred. Otherwise they learn as the best model does (the same rules, digit-sum concepts and narrow-favouring, open-ended-discounted, margin-credited ranges, with strays and guessing). This predicts that broad rules and wide ranges consistent with the examples keep a floor of "yes" that does not shrink as more examples are added, e.g. more "yes" to far members of a wide consistent range or to other even numbers after 16, 8, 64 than the pure size principle gives.

### prototype_feature_ratio_similarity — pruned (experiment2 end of experiment)

**Outcome:** 252.2 nats behind margin_and_heading_range_prior (10.8× dse)

**Hypothesis:** People judge the target by its similarity to the example set's feature prototype, where the features are number properties (arithmetic and digit properties, digit sums, and lying within a given stretch of the number line), each feature counts by the share of examples that have it, and rarer properties count as more distinctive. Similarity is the proportion of the prototype's distinctive feature mass that the target shares, and the more similar the target, the more likely a "yes" (with occasional guessing). Because feature salience is fixed and there is no inference over hidden programs, adding examples does not sharpen generalization beyond the properties they share, so after 16, 8, 64 people should say "yes" to members of broad shared properties (e.g. even or small numbers) more often than a size-principle learner predicts.

### centred_examples_range_prior — pruned (experiment2 end of experiment)

**Outcome:** 2.2 nats behind margin_and_heading_range_prior; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 26

**Hypothesis:** Refinement of the incumbent `margin_beyond_examples_range_prior`: people do Bayesian concept learning over the same arithmetic and digit rules, the 18 digit-sum concepts and every numeric range of 1-100. They read the examples as distinct draws with some strays, sometimes guess, favour narrow ranges, discount open-ended ranges that run to 1 or 100, and credit a range by how far it reaches past the examples. The one change is a centring expectation: people expect the examples they were shown to sit in the middle of the program's range rather than off to one side of it, so a candidate range that leaves much more room on one side of the examples than on the other is held less plausible, in proportion to how lopsided that room is (the strength is inferred; the data suggest a penalty for lopsided ranges). This predicts that "yes" spreads evenly around a cluster. Numbers just beyond one end get "yes" mainly through ranges that also reach past the other end, so the far reaches on one side only (e.g. 75 after 60, 62, 64) get fewer "yes" than the incumbent gives, while numbers a little beyond either end keep their credit. The effect does not depend on the order the examples were shown in, and rule and digit-sum generalization are unchanged.

### size_principle_for_rules_only — pruned (experiment2 end of experiment)

**Outcome:** 79.0 nats behind margin_and_heading_range_prior (7.3× dse)

**Hypothesis:** People apply the size principle only to symbolic rules: for arithmetic, digit and digit-sum rules a small rule that fits every example is a suspicious coincidence, but for stretches of the number line an example is just a number the range happens to contain, so every range covering the examples stays credible in proportion to its prior plausibility alone (the best model's narrow-favouring, open-ended-discounted, margin-credited range prior), however wide it is and however many examples are shown. This predicts that magnitude-based generalization does not tighten as more examples fall in a cluster (about as much "yes" to 70 after 60, 62, 64 as after 60 alone), whereas rule-based generalization sharpens with each example as before.

### decade_named_concepts — rejected (experiment3 round 0 candidate 1 lens 9)

**Outcome:** predicts like existing model 'heading_direction_range_extrapolation' (p_yes RMSE 0.00046 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_direction_range_extrapolation, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100 as the best model (distinct examples with some strays, occasional guessing, ranges stretched toward where the sequence is heading). They also hold one more kind of named concept: the decades, "numbers in the twenties", "in the eighties", and so on, plus "the single-digit numbers". All ten decades are equally likely a priori, and the family as a whole takes an inferred share of the prior belief given to rules. Because a decade is a named stretch of the number line, this predicts that examples falling inside one decade (e.g. 83, 87 or 24, 28) draw more "yes" to the rest of that decade and a sharper drop just past its round-number boundaries than a learner who weighs all ranges by width alone.

### decade_named_concept_boost — rejected (experiment3 round 0 candidate 1 lens 9 repair 1)

**Outcome:** predicts like existing model 'heading_direction_range_extrapolation' (p_yes RMSE 0.00024 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_direction_range_extrapolation, not a new hypothesis.

**Hypothesis:** People do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and numeric ranges as the best model (distinct examples with some strays, occasional guessing, ranges stretched toward where the sequence is heading), but they also hold the ten decades ("the single-digit numbers", "the twenties", ..., "the nineties") as named concepts, all equally likely, with a stated prior: naming a decade adds about one-sixth to the prior plausibility of that stretch of the number line, beyond what an unnamed range of the same width gets. This predicts that examples inside one decade (e.g. 83, 87 or 24, 28) draw somewhat more "yes" to the rest of that decade and a sharper drop just past its round-number boundaries (e.g. fewer "yes" to 91 after 83, 87) than a learner who weighs ranges by width alone.

### set_size_triggered_rule_search — rejected (experiment3 round 1 candidate 2 lens 2)

**Outcome:** predicts like existing model 'heading_extrapolation_weber_width' (p_yes RMSE 0.00168 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_extrapolation_weber_width, not a new hypothesis.

**Hypothesis:** People only start looking for an arithmetic or digit rule when there is a pattern across several examples to notice: with a single example they think mainly in terms of a stretch of the number line around it, and each further example shown makes a rule (e.g. "powers of 2", "ends in 3", digits adding to 9) come to mind more readily relative to a numeric range, with the strength of this shift inferred. Otherwise they do Bayesian concept learning as the best model does (distinct examples with some strays, occasional guessing, Weber-like range breadth, ranges stretched toward where the sequence is heading). This predicts weaker rule-based generalization after one or two examples (e.g. fewer "yes" to 32 after 16, or to 64 after 16, 8) and stronger after three or four than a learner whose rule-versus-range prior ignores how many examples were shown.

### upward_reach_weber_width — rejected (experiment3 round 1 candidate 4 refine incumbent heading_extrapolation_weber_width)

**Outcome:** predicts like existing model 'heading_weber_with_upward_counting' (p_yes RMSE 0.00079 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_weber_with_upward_counting, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `heading_extrapolation_weber_width`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays, sometimes guess, and judge a range's breadth Weber-like, relative to the size of the numbers it covers. The one change, the component taken from `upward_room_range_prior`, replaces the incumbent's order-dependent heading stretch: people expect the program's numbers to carry on upward past the largest example, as counting does, whatever order the examples were shown in, so they credit a candidate range by how far it reaches above the examples (strength inferred). This predicts more "yes" just above a cluster than just below it for the same set in rising or falling order (e.g. 67 versus 57 after 60, 62, 64 and after 64, 62, 60), with the incumbent's magnitude-relative tightening of generalization around large clustered examples kept.

### upward_room_weber_width — rejected (experiment3 round 1 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'heading_weber_with_upward_counting' (p_yes RMSE 0.00102 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_weber_with_upward_counting, not a new hypothesis.

**Hypothesis:** Refinement of `upward_room_range_prior`: people do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100, reading the examples as distinct draws with some strays, sometimes guessing, discounting open-ended ranges, and expecting the program's numbers to carry on upward past the largest example (crediting a range by how far it reaches above the examples, whatever the order shown). The one change, the component taken from the incumbent `heading_extrapolation_weber_width`, is that people judge a range's breadth Weber-like, by its width relative to the magnitude of the numbers it covers rather than its absolute width, so "60 to 70" counts as a more specific range than "2 to 12". This predicts an order-free upward spread of "yes" above clusters whose falloff is tighter around large examples than around small ones, while rule and digit-sum generalization are unchanged.

### heading_weber_two_sided_margin — rejected (experiment3 round 1 candidate 4 refine incumbent heading_extrapolation_weber_width repair 1)

**Outcome:** predicts like existing model 'heading_weber_with_upward_counting' (p_yes RMSE 0.00198 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of heading_weber_with_upward_counting, not a new hypothesis.

**Hypothesis:** Refinement of the incumbent `heading_extrapolation_weber_width`. People do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays and sometimes guess, judge a range's breadth Weber-like (relative to the magnitude of the numbers it covers), and let the direction of the last step stretch the ranges toward where the sequence is heading. The one change, the component taken from `margin_beyond_examples_range_prior`, is a two-sided margin expectation. People do not expect the few numbers they were shown to sit right at the edges of the program's range, so they credit a candidate range by how far it reaches past the examples on both sides, below the smallest as well as above the largest, whatever order the examples came in (strength inferred). This predicts more "yes" a little beyond either end of a cluster (e.g. both 57 and 67 after 60, 62, 64) than the incumbent gives, tilted by the heading. It also predicts more "yes" just below a cluster than an upward-only counting habit gives.
