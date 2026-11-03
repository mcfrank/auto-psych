# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## equal_prior_concepts_with_guess_lapse  — rank 0, the best model on this data, ELPD-LOO -908.2

Refinement of `concept_learning_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle and 10% of examples random, and on a fixed fraction of trials they ignore the concept and guess "yes" at their own baseline rate — but they hold no learned preference among the rules: every rule gets the same prior weight. The one change is replacing the 45 separately inferred rule weights with a single shared rule prior, which claims that the lapse model's advantage comes from occasional guessing, not from idiosyncratic rule preferences, so the model should generalize as well with 43 fewer free parameters.

## digit_concepts_with_guess_lapse  — rank 1, 7.0 ± 5.6 nats behind the best (1.3× dse: statistically tied with the best), ELPD-LOO -915.3

Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning over Tenenbaum's mathematical rules, the digit concepts ("contains the digit d", "ends in 0") and size-weighted intervals exactly as in that model, but on a small fixed fraction of trials they do not consult the inferred concept and instead guess, saying "yes" at their own baseline rate. The one change is this response-lapse component (taken from `concept_learning_with_guess_lapse`), which predicts occasional "no" answers to targets the concept surely contains and occasional "yes" answers to targets it surely excludes, instead of near-certain responses there.

## tenenbaum_digit_concepts  — rank 2, 10.0 ± 8.1 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -918.2

Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over Tenenbaum's mathematical rules and size-weighted intervals, but they also read the examples as written numerals, so their hypothesis space additionally contains digit concepts — "contains the digit d" (for each digit 0–9) and "ends in 0" — sharing the rules' equal prior. The one change is adding these digit concepts, which predicts that examples sharing a digit (e.g. 83, 38, 58, 68) lead people to say yes to other numbers containing that digit (e.g. 84, 87), even when those numbers are far from any arithmetic rule.

## digit_concepts_inferred_outlier_rate  — rank 3, 11.3 ± 8.0 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -919.5

Refinement of `tenenbaum_digit_concepts`: people do Bayesian concept learning with the size principle over Tenenbaum's arithmetic rules, size-weighted intervals and the digit concepts ("contains the digit d", "ends in 0"), while allowing that any example may be a stray random number from 1–100 — and how often they believe the program produces such strays is a property of the learner, inferred from their answers, rather than fixed at one in ten. The one change is freeing this outlier rate, which predicts that people discount an example that fails to fit a tight concept as a stray rather than abandoning the concept for a broader one, so narrow rules survive a misfitting example and generalization to their members stays strong.

## order_weighted_concept_learning  — rank 4, 12.5 ± 8.2 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -920.8

People do Bayesian concept learning over mathematical rules, digit concepts and numeric intervals (the size principle), but they read the examples one after another and do not weigh them equally: their attention to an example shifts systematically with its position in the sequence, so later-read examples (recency) or earlier-read examples (primacy) count for more of the evidence about the concept. The one claim is this order-dependent weighting of each example's evidence — the total amount of evidence stays the same — which predicts that the same set shown in a different order generalizes more toward the concepts favoured by the emphasized examples.

## conservative_evidence_digit_concepts  — rank 5, 14.4 ± 7.3 nats behind the best (2.0× dse: statistically tied with the best), ELPD-LOO -922.6

People do Bayesian concept learning over Tenenbaum's mathematical rules, digit concepts and size-weighted intervals, but they are conservative updaters: they underweight (or overweight) the evidence each example provides, so the likelihood of the examples under each concept enters their belief only to a fitted power rather than at full strength. This single mis-weighting predicts that generalization tightens toward the narrowest consistent concept more slowly (or faster) with added examples than an ideal size-principle learner would, so multi-example sets still yield graded "yes" to broader consistent concepts.

## concept_learning_with_guess_lapse  — rank 6, 17.1 ± 3.6 nats behind the best (4.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -925.3

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules with separately weighted priors plus every interval of 1-100 (size principle, 10% of examples random), but on a fixed fraction of trials they do not consult the inferred concept at all and instead answer by guessing, saying "yes" at their own baseline rate. The one change is this response-lapse component, which lets the model explain "no" answers to targets the concept clearly contains (even the shown examples) and stray "yes" answers to targets it clearly excludes.

## bigelow_piantadosi_independent  — rank 7, 18.3 ± 4.8 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -926.6

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## smallest_rule_commitment  — rank 8, 195.9 ± 19.6 nats behind the best (10.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1104.2

People do not weigh many candidate concepts at once; they search for the single most specific rule that covers every example (the smallest of the arithmetic and digit rules such as "powers of 2", "multiples of 7", "ends in 3" or "contains an 8"), commit to it, and answer yes to a target if and only if it obeys that rule, falling back to the range spanned by the examples only when no rule fits them all. Apart from response noise (a fixed rate of yes inside the chosen rule and of yes outside it), generalization is all-or-none with no graded similarity or averaging over alternative concepts.

## weak_sampling_concept_generalization  — rank 9, 209.9 ± 20.6 nats behind the best (10.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1118.2

People entertain the same mathematical rules and numeric ranges as candidate concepts, but treat the examples only as labelled members of the concept rather than as random draws from it (weak sampling, no size principle): every concept consistent with the examples stays credible in proportion to its prior, however broad it is, so adding examples rules concepts out but never sharpens generalization toward the narrowest one. The sharpest disagreement with the best (strong-sampling) model is on multi-example sets that fit both a narrow rule and broader ones (e.g. 16,8,64 or 83,38,58,68): this model predicts broad, graded "yes" to targets in wide consistent ranges and broad rules, where the size-principle model predicts confident yes only for the narrow rule.

## exemplar_feature_similarity_gcm  — rank 10, 304.9 ± 28.0 nats behind the best (10.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1213.2

People do not entertain explicit rules; they judge the target by its summed psychological similarity to the remembered examples (an exemplar / generalized-context model). Similarity decays exponentially with distance in a feature space where each number is described by its magnitude and by its surface properties (parity, last digit, tens digit, divisibility by 3/4/5, being prime or a square), with attention split between magnitude and properties; the more similar the target is to the examples on average, the more likely a "yes".

## digit_feature_exemplar_similarity  — rank 11, 323.8 ± 29.9 nats behind the best (10.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1232.1

People judge a target by its featural resemblance to the examples in decimal notation: each example's similarity to the target falls off exponentially with whether the two differ in their ones digit and whether they differ in their tens digit, and the probability of saying yes grows with the target's summed (averaged) similarity to the shown examples, as in an exemplar model over digit features rather than a search over explicit rules.

## tenenbaum_rules_similarity  — rank 12, 340.8 ± 41.9 nats behind the best (8.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1249.1

Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

## first_example_anchored_concept_learning  — no comparison row

People do Bayesian concept learning over the same arithmetic rules, digit concepts and numeric intervals as the best model (size principle, equal rule prior, occasional guessing), but they build their guess about the concept from the first example they read and treat it as a sure member of the concept: only the examples that follow can be dismissed as stray random numbers. This primacy-of-the-anchor claim predicts that a set whose first example misfits a tight rule shared by the later ones generalizes broadly (the rule must include the anchor), whereas the same misfit shown last is discounted and the tight rule survives.

## spread_adaptive_magnitude_gradient  — no comparison row

People judge a target by its numerical closeness to the nearest shown example, with an exponentially decaying (Shepard) generalization gradient whose width is not fixed but scales with how spread out the examples are: tightly clustered examples (e.g. 60, 62, 64) produce a sharp gradient that says "yes" only to nearby numbers, while widely scattered examples produce a broad gradient that extends "yes" to distant numbers. The single claim is this spread-adaptive magnitude similarity, with no rules or digit features.

## soft_confidence_criterion_decision  — no comparison row

People do Bayesian concept learning exactly as in the simplest digit-concept model (equal-prior mathematical and digit rules plus size-weighted intervals), but they do not answer "yes" with the probability that the concept contains the target; instead they compare their degree of belief against a personal criterion of how sure they must be, answering "yes" when their confidence that the target belongs exceeds that criterion, with graded (noisy) sensitivity around it. This single decision-threshold rule predicts that moderately probable targets are pushed toward "no" (or "yes") depending on where the criterion sits, and that near-certain beliefs are softened or sharpened by how steeply responses track the confidence-versus-criterion gap.

## iter2_candidate3  — no comparison row

Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 rules (including the digit concepts) and every interval of 1-100, with the size principle, 10% of examples random, and a guess lapse at their own baseline "yes" rate — but the rules are not all equally salient: each rule's prior weight is a draw from a shared population of rule saliences whose spread is inferred, so a rule departs from the common weight only as far as the answers demand. The one change is replacing the single shared rule prior with this partially pooled (hierarchical) rule prior, which claims that some rules (e.g. "odd numbers") are systematically less or more attractive than others, yet predicts generalization closer to the equal-prior model than 45 freely inferred weights would.

## rule_breadth_prior_concepts  — no comparison row

Refinement of `equal_prior_concepts_with_guess_lapse`: people do Bayesian concept learning over the same 45 arithmetic and digit rules and every interval of 1-100 (size principle, 10% of examples random), with occasional guessing at their own yes rate — but before seeing any examples they already prefer rules by their breadth, giving each rule a prior weight that rises or falls as a power of how many numbers it contains (a preference for specific, "interesting" rules over broad ones, or the reverse). The one change is replacing the flat prior across rules with this single size-graded rule prior, which predicts that broad rules such as "odd" or "even" win less (or more) of the posterior than an equal-prior learner gives them, e.g. fewer "yes" answers to arbitrary odd numbers after 85, 19, 91.
