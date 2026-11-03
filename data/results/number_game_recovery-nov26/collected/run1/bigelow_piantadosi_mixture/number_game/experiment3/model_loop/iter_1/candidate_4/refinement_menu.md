# Refinement menu

The models you may refine, other than the incumbent `evidence_graded_misfit_tolerance`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### misfit_tolerant_coincidence_rethink — rank 1, 1.0 ± 1.0 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3126.9

**Hypothesis:** Refinement of the incumbent coincidence_triggered_guess_rethink (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and made salient only once a digit is shared by two examples, a newly noticed coincidence triggering a rethink, fitted trust in the examples). The one change: a guess is not automatically abandoned when a new example contradicts it — people sometimes write the misfitting example off as a stray number and keep their guess, with a fitted tolerance. So after 77 and 17 a guess of "ends in 7" often survives the outlier 8, and targets fitting the early examples' concept are still endorsed despite a later example that breaks it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/misfit_tolerant_coincidence_rethink.py`

### coincidence_triggered_guess_rethink — rank 2, 1.3 ± 1.5 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3127.2

**Hypothesis:** Refinement of the incumbent digit_coincidence_noticing_guess (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted and made salient only once a digit is noticed as shared by at least two examples, fitted trust in the examples). The one change: noticing a coincidence is itself a trigger to rethink. When a new example makes people notice a digit it shares with an earlier example, they drop their current guess with a fitted probability and reconsider the concept from all the examples, even if that guess still fits. So after 46 then 86, a range guess formed from 46 that happens to contain 86 is often given up in favour of "ends in 6", and far numbers like 96 are endorsed more than the purely sticky learner predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_triggered_guess_rethink.py`

### compressed_concept_size_sticky_guess — rank 3, 2.8 ± 1.9 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -3128.7

**Hypothesis:** People learn the concept as in the current best model (a single sticky working guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, with the size principle favouring small concepts), but with one distortion: they misjudge how many numbers a candidate concept contains, perceiving concept size on a compressed, numerosity-like scale relative to the whole range 1-100 (with the compression fitted). So they feel "powers of 2" or a narrow range is less small, and "even numbers" less large, than it really is, which weakens (or, if expanded, strengthens) the size principle's preference for specific concepts without changing how strongly a non-fitting example counts against a guess.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/compressed_concept_size_sticky_guess.py`

### graded_digit_salience_guess — rank 4, 4.1 ± 1.8 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3130.0

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples, and rules about how numbers are written made salient only by a noticed shared digit). The one change: noticing a shared digit is not all-or-none at two examples but grows with every example that carries it — the reduced plausibility of a written-form rule ("ends in 6", "contains a 3") is divided among the examples that have its digit, so it is fully reduced with one such example, half as reduced with two, a third with three — making "ends in 6" a gradually more salient guess as more examples repeat the digit, rather than fully salient as soon as two do.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/graded_digit_salience_guess.py`

### coincidence_salience_guess_revision — rank 5, 6.1 ± 2.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3132.0

**Hypothesis:** Refinement of coincidence_gated_all_rules_guess (a single sticky working guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, with rules made salient only once two examples share them). The one change: noticing a coincidence is itself a trigger for rethinking — when a new example makes a rule shared by two examples for the first time, people abandon their current guess and reconsider the concept from all examples (even if the guess still fits) with a probability equal to how compelling the newly noticed rule is given the examples so far, with no extra free parameter. So a striking pattern revealed by later examples overrides a guess formed from an unrepresentative first example, while a weak coincidence barely unsettles it, and order effects are weaker than pure stickiness predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_salience_guess_revision.py`

### coincidence_gated_all_rules_guess — rank 6, 6.1 ± 2.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3132.1

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning and kept with fitted stickiness while later examples fit it, digit rules discounted, fitted trust in the examples, and digit rules made salient only once a digit is noticed as shared by at least two examples). The one change: the same coincidence-noticing gate applies to arithmetic rules too — "even", "multiples of 7" or "squares" becomes a salient candidate only once at least two of the examples seen so far share that property, and until then (always after a single example) its plausibility is reduced by its own fitted availability factor. So a lone example suggests a range of nearby numbers rather than a broad rule, people generalise less to far-away numbers from one example, and the first working guess is range-like until a shared property is noticed.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_gated_all_rules_guess.py`

### coincidence_noticing_guess_vote — rank 7, 7.1 ± 3.0 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3133.0

**Hypothesis:** Refinement of digit_coincidence_noticing_guess (a single sticky working concept guess — a rule or a numerical range — formed from the first example by Bayesian concept learning, digit rules discounted and made salient only once a digit is noticed as shared by two examples, fitted trust in the examples). The one change, the component taken from sampled_guess_vote_threshold: people do not answer by consulting a single guess, but call to mind several guesses generated by this learner and say "yes" only if a sufficient share of them contain the target, with the effective number of guesses and the share required fitted. So the same belief maps onto a sharpened or softened, more liberal or more conservative yes rate — e.g. targets that only some plausible guesses contain (96 after 46 and 86) can be endorsed less, and near-certain members more, than probability matching predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_noticing_guess_vote.py`

### compressed_line_category_exemplars — rank 8, 50.8 ± 12.1 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3176.7

**Hypothesis:** People do not infer the program's concept; they retrieve familiar number categories (rules such as "even" or "powers of 2", and ranges bounded by multiples of five) stored in memory as lists of their member numbers, activating each category by how similar the examples are to its stored members, and endorse the target by its similarity to the nearest member of the activated categories. The comparison is made on a compressed mental number line (compression fitted), so the same numerical gap counts as less dissimilar among large numbers than among small ones: generalisation from large examples (e.g. 82, 92) spreads further than from small ones (e.g. 2, 12), and small categories crowded among small numbers are matched more sharply.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/compressed_line_category_exemplars.py`

### noisy_criterion_belief_threshold — rank 9, 65.7 ± 8.3 nats behind the best (7.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3191.6

**Hypothesis:** People infer the concept as the simplest current sticky-guess learner does, but they do not answer by probability matching: they compare how strongly they believe the target belongs against a fixed internal criterion of how sure they must be, and that comparison is perturbed by trial-to-trial noise in the felt belief, so they say "yes" with a probability that rises smoothly (logistically) with belief strength around the criterion. This noisy criterion produces both a ceiling below 1 for certain members (the examples themselves) and a floor above 0 for clear non-members, and makes moderately believed targets fall clearly on one side of the criterion rather than being endorsed in proportion to belief.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/noisy_criterion_belief_threshold.py`

## Pruned models (out of the set; narrowest margin first)

### sticky_guess_fitted_sampling_trust — pruned (experiment2 end of experiment)

**Margin:** 2.9 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 20

**Hypothesis:** Refinement of digit_rule_discount_sticky_guess: people hold a single working guess about the concept (a rule or a numerical range, with a discounted prior on digit-based rules and power-law/Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted. The one change (the component taken from inferred_sampling_reliability_concepts): how much people trust that the examples were really drawn from the program's concept, rather than being stray numbers, is fitted instead of fixed at 90% — so the strength of the size principle, and hence how readily people entertain (and answer by) guesses that do not even contain every example, is set by the data.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sticky_guess_fitted_sampling_trust.py`

### range_strain_triggered_rule_search — pruned (experiment2 end of experiment)

**Margin:** 3.1 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 20

**Hypothesis:** People look for a numerical range first and search for a rule only when a range is a strained explanation: the more widely the examples seen so far are spread over the number line, the more readily they consider rules ("powers of 2", "even", "ends in 6"), whereas a single example or a tight cluster of examples is read as a place on the number line and rules stay largely unconsidered (by a fitted availability factor). Otherwise they hold a single sticky working guess formed from the first example by Bayesian concept learning, as in the current best model — so clustered examples that share a rule (20, 22, 24) are generalised as a range, while scattered ones (16, 8, 64) prompt the rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_strain_triggered_rule_search.py`

### coincidence_gated_rule_guess — pruned (experiment2 end of experiment)

**Margin:** 3.2 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 20

**Hypothesis:** People do not suspect a rule of any kind — arithmetic ("powers of 2", "multiples of 7", "squares") or written ("ends in 6") — until it is suggested by a coincidence: a rule becomes a salient candidate only once at least two of the examples seen so far satisfy it, and before that its plausibility is reduced by a fitted availability factor, so a lone example is generalised mostly as a numerical range around it. Otherwise they hold a single sticky working guess formed from the first example by Bayesian concept learning, as in the current best model (whose availability penalty applies only to digit rules). The sharpest disagreement lives in single-example sets and in sets where only one example fits an arithmetic rule: after a lone 16 the best model still endorses far-away 64, 4 or 32 as powers of 2, whereas this model predicts these far rule-consistent targets are rejected until a second example such as 8 makes the rule a noticed coincidence.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/coincidence_gated_rule_guess.py`

### digit_rule_discount_sticky_guess — pruned (experiment2 end of experiment)

**Margin:** 3.3 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 20

**Hypothesis:** Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted, with range guesses shaped by a power-law endpoint and Erlang width prior. The one change: rules defined only by how a number is written ("ends in 7", "contains the digit 3") are a priori less (or more) plausible than arithmetic rules (odd, multiples, powers, squares, primes), by a single fitted weight on the digit-rule family — so people latch onto and extend "same last digit" guesses to far-away numbers less readily than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/digit_rule_discount_sticky_guess.py`

### target_as_evidence_sticky_guess — pruned (experiment2 end of experiment)

**Margin:** 3.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 20

**Hypothesis:** Refinement of digit_rule_discount_sticky_guess (Bayesian concept learning over rules and numerical ranges, a single sticky working guess formed from the first example, digit rules discounted). The one distortion: people treat the question itself as weak evidence — they partly assume the experimenter tends to ask about numbers the program actually produces, so the asked-about target is counted, with a fitted fraction of the weight of a real example, as one more sample from the concept. This "leading question" bias inflates "yes" for targets that would belong to a small, specific concept (e.g. a power of 2 after 16 and 8) far more than for targets that only fit broad concepts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/target_as_evidence_sticky_guess.py`

### sampled_guess_vote_threshold — pruned (experiment2 end of experiment)

**Margin:** 3.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 20

**Hypothesis:** People form guesses about the concept exactly as in the simplest sticky-guess learner (a working guess from the first example, kept while later examples fit it, revised from all examples when contradicted), but they do not answer by reporting one guess: they call to mind a handful of such candidate guesses and say "yes" only if a sufficient share of them contain the target. The effective number of guesses consulted and the share required (a yes/no criterion) are fitted, so the same belief maps onto a sharpened or softened, more liberal or more conservative yes rate; one guess with no criterion recovers plain probability matching.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sampled_guess_vote_threshold.py`

### erlang_interval_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 3.6 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule), but they do not regard every interval as equally plausible a priori. Instead, their prior over intervals follows an Erlang density on the interval's size with a fitted characteristic size, so very narrow ranges hugging the examples are a priori implausible and ranges of a typical intermediate width are favoured; this one change controls how far beyond the examples' span people generalise when they think in terms of a numerical range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_prior_concepts.py`

### first_guess_win_stay_concepts — pruned (experiment2 end of experiment)

**Margin:** 4.2 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 20

**Hypothesis:** People do not weigh all examples together at once; they hold a single working guess about the concept (a rule or a numerical range) that they form from the first example by Bayesian concept learning, and as each further example arrives they keep that guess, with a fitted stickiness, as long as the new example fits it, reconsidering the concept from all examples seen so far only when the new example contradicts the guess (or when they do not stick). Because the guess is anchored on what fitted the early examples, a broad concept chosen after the first example survives later examples that would have favoured a narrower one, so the same examples in a different order produce different generalisations (a primacy effect).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_guess_win_stay_concepts.py`

### sticky_guess_last_digit_rule_prior — pruned (experiment2 end of experiment)

**Margin:** 4.3 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 20

**Hypothesis:** Refinement of sticky_guess_power_law_ranges: people hold a single working guess about the concept (a rule or a numerical range, with power-law and Erlang priors on ranges), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering only when contradicted. The one change: rules are not all equally natural guesses — the ten "ends in digit d" rules share a single fitted prior weight relative to the other rules, so people latch onto and keep last-digit rules less (or more) readily than arithmetic rules like multiples or powers, which governs how much they endorse far-away numbers merely because they share an example's final digit.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sticky_guess_last_digit_rule_prior.py`

### sticky_guess_power_law_ranges — pruned (experiment2 end of experiment)

**Margin:** 4.4 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 20

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, as later examples arrive as long as each new example fits it, reconsidering from all examples so far only when it is contradicted (or they do not stick). The one change, the component taken from power_law_interval_endpoints: when they consider a numerical range they do not treat all locations on the number line alike — the range's endpoints have a prior plausibility that follows a fitted power law over 1-100 (on top of the incumbent's Erlang preference for a typical width), so whether a range guess is formed and kept depends on whether the examples are small or large numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sticky_guess_power_law_ranges.py`

### sticky_guess_rule_specific_priors — pruned (experiment1 end of experiment)

**Margin:** 4.5 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 21

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning, keep it with a fitted stickiness while new examples fit it, and reconsider from all examples seen so far only when an example contradicts it (or they do not stick), producing a primacy effect. The one change is that rules are not equally plausible a priori: each rule (e.g. "powers of 2" versus "ends in 7") has its own fitted prior weight (the component taken from bigelow_piantadosi_independent / erlang_interval_prior_concepts), so which guess people first latch onto — and therefore which guess survives later examples — depends on how natural each rule is to them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sticky_guess_rule_specific_priors.py`

### fuzzy_trace_example_memory — pruned (experiment2 end of experiment)

**Margin:** 4.7 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 20

**Hypothesis:** People do Bayesian concept learning over mathematical rules and numerical ranges (with the size principle), but their memory of the examples is imprecise: each example is kept verbatim only with a fitted probability, and otherwise only its rough magnitude (within about five of its true value) is retained. Concepts are evaluated against these fuzzy memory traces, so concepts that depend on an example's exact identity (last digit, multiples, powers) lose support with every imperfectly remembered example while numerical ranges, which only need the magnitude, do not — making people generalise less along digit-based rules to far-away targets, especially from longer sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/fuzzy_trace_example_memory.py`

### range_restricted_rule_concepts — pruned (experiment1 end of experiment)

**Margin:** 4.9 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 21

**Hypothesis:** Refinement of power_law_endpoints_erlang_width (same Bayesian concept learning over the 45 rules and all intervals of 1-100, size principle, 10% noisy examples, a prior weight per rule, power-law endpoint and Erlang width priors on ranges). The one change: people can also entertain a rule restricted to a range of the number line — e.g. "numbers ending in 3 between 10 and 40" rather than "all numbers ending in 3" — with a fitted share of each rule's prior given to its range-restricted version (any range equally plausible). Because such a restricted rule is smaller, the size principle favours it when the examples cluster, so people generalise a rule (especially a surface one like a shared last digit) less to numbers far outside the examples' span than the unrestricted rule predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_restricted_rule_concepts.py`

### sticky_guess_attentional_lapse — pruned (experiment2 end of experiment)

**Margin:** 4.9 nats behind coincidence_triggered_guess_rethink (2.6× dse)

**Hypothesis:** Refinement of sticky_guess_fitted_sampling_trust: people hold a single working guess about the concept (a rule or a numerical range, digit rules discounted, power-law/Erlang priors on ranges, fitted trust in the examples), formed from the first example by Bayesian concept learning and kept with a fitted stickiness while later examples fit it, reconsidering from all examples only when contradicted. The one change (the component taken from lapse_biased_guess_concepts): on a fitted fraction of trials people do not consult their guess at all and simply guess, answering "yes" with a fitted bias — which explains why they reject even the examples themselves about a quarter of the time without forcing the concept learner to distrust its examples, and puts a common floor under "yes" for numbers outside every plausible concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/sticky_guess_attentional_lapse.py`

### one_back_digit_coincidence_guess — pruned (experiment2 end of experiment)

**Margin:** 4.9 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 20

**Hypothesis:** People hold a single working guess about the concept (a rule or a numerical range), formed from the first example and kept while later examples fit it, and a rule about how numbers are written ("ends in 6", "contains the digit 3") becomes a salient guess only once they notice its digit as a coincidence. They notice such coincidences by a one-back working-memory comparison, setting each new example against the example shown just before it, so a shared digit is noticed only when two consecutive examples have it. So the order of the examples matters: "46, 86, 13" makes "ends in 6" salient and leads people to say yes to 96, whereas "46, 13, 86" does not, although both contain the same numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/one_back_digit_coincidence_guess.py`

### power_law_interval_endpoints — pruned (experiment1 end of experiment)

**Margin:** 5.0 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule), but when they imagine the program picking a numerical range they do not treat every endpoint as equally likely: the two endpoints are drawn from a power-law distribution over 1-100 whose exponent is fitted, so a range's prior plausibility depends on where on the number line it lies (among small versus large numbers), not just on whether it exists. This magnitude-dependent prior on interval endpoints is the one change; an exponent of zero recovers the incumbent's uniform endpoints. It changes how strongly range concepts compete with rules for small-number versus large-number examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/power_law_interval_endpoints.py`

### similarity_route_erlang_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.3 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 21

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but on a fitted fraction of judgements they do not consult the inferred concept at all and instead answer by number-line similarity, saying "yes" more the closer the target is to the nearest example (a component taken from the exemplar-similarity models). The one change is this similarity route: it makes people endorse far-away numbers that fit a rule only through a surface property (such as sharing an example's last digit) less than pure concept learning predicts, and near neighbours of the examples more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/similarity_route_erlang_concepts.py`

### decade_restricted_rule_concepts — pruned (experiment1 end of experiment)

**Margin:** 5.6 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 21

**Hypothesis:** People do Bayesian concept learning over familiar mathematical rules and numerical ranges (size principle, a prior weight per rule, an Erlang prior on range width), but they can also entertain a rule confined to a region of the number line bounded by decades — e.g. "numbers ending in 3 between 10 and 50" or "even numbers in the twenties" — with a fitted share of each rule's prior going to such restricted versions. Because a restricted rule is smaller, the size principle favours it whenever the examples all lie in one region, so people extend a rule like "ends in d" or "multiples of m" to far-away numbers less than a pure-rule learner does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_restricted_rule_concepts.py`

### erlang_width_interval_concepts — pruned (experiment1 end of experiment)

**Margin:** 6.5 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 21

**Hypothesis:** Refinement of bayesian_concepts_logodds_readout: people do the same Bayesian concept learning over the 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples) and answer through the same scaled, shifted log-odds readout, but they do not consider all interval sizes equally plausible a priori. The one change: an interval's prior depends on its width through an Erlang density with a fitted typical width, so people expect the program's range of numbers to have a characteristic breadth, which sets how far they generalise beyond the span of the examples independently of the size principle.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_width_interval_concepts.py`

### recency_weighted_guess_revision — pruned (experiment2 end of experiment)

**Margin:** 6.5 nats behind coincidence_triggered_guess_rethink; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 20

**Hypothesis:** Refinement of first_guess_win_stay_concepts: people hold a single working guess about the concept (a rule or a numerical range), formed from the first example by Bayesian concept learning and kept, with a fitted stickiness, while each new example fits it. The one change is in how they revise when a new example contradicts the guess (or they do not stick): they rebuild the guess around the newest example, which they weigh fully, while their recall of the earlier examples is only partial — earlier examples' evidence is down-weighted by a fitted retention factor — so revisions are recency-driven and the final concept is pulled toward whatever fits the example that forced the change.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/recency_weighted_guess_revision.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 6.7 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 16 of 21

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### pedagogical_teacher_sampling — pruned (experiment1 end of experiment)

**Margin:** 7.0 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 17 of 21

**Hypothesis:** People treat the examples as having been chosen on purpose by a helpful teacher (pedagogical sampling): when they ask how likely an example is under a candidate concept (a rule such as "powers of 2" or a numerical range), they expect a teacher who held that concept to pick examples from it that are diagnostic of it — numbers that would lead a naive learner to that concept rather than to others — and not just any member at random. Bayesian concept learning over rules and ranges with this teacher-based likelihood, whose strength of pedagogical expectation is fitted (zero recovers ordinary random sampling from the concept), sets how people generalise to a target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/pedagogical_teacher_sampling.py`

### bayesian_concepts_logodds_readout — pruned (experiment1 end of experiment)

**Margin:** 7.1 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 18 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (size principle, 10% noisy examples), but they do not report the posterior probability of membership directly; instead they answer "yes" through a graded, possibly biased decision readout in which the log-odds of membership is scaled by a free confidence factor and shifted by a free yes/no criterion. This lets near-certain posteriors (e.g. for the examples themselves, or for numbers far outside the concept) map onto less extreme, empirically observed yes rates, and lets judges be more or less liberal than the posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bayesian_concepts_logodds_readout.py`

### inferred_sampling_reliability_concepts — pruned (experiment1 end of experiment)

**Margin:** 7.4 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 19 of 21

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100, with the size principle, but how strongly they believe the examples were drawn from the concept (rather than at random from 1-100) is a free parameter inferred from the data instead of being fixed at 90%. This one change sets how sharply the size principle favours small, specific concepts over broad ones, and so how tightly people generalise from a few examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_sampling_reliability_concepts.py`

### range_bounded_rule_concepts — pruned (experiment1 end of experiment)

**Margin:** 7.9 nats behind sticky_guess_power_law_ranges (2.4× dse)

**Hypothesis:** People's hypothesis space contains not only whole-number-line rules ("even numbers", "ends in 7") and numerical ranges, but also rules confined to a range ("even numbers in the 20s", "multiples of 5 below 40"); because such range-bounded rules are smaller, the size principle favours them, so people generalise a rule mainly to targets near the examples. The disagreement with the current best model lives in targets far outside the examples' range that fit a rule only by a property such as their last digit or a common multiple: the best model endorses them strongly, this one predicts that endorsement falls off with distance from the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_bounded_rule_concepts.py`

### leaky_memory_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 8.3 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 20 of 21

**Hypothesis:** People do Bayesian concept learning over mathematical rules and intervals (with the size principle), but they read the examples one at a time and their memory of earlier examples fades: each example's evidence is discounted by a free forgetting factor for every example shown after it, so the most recent example counts fully and the first one of a long set counts least. As a result, concepts that fit only the later examples (and rules or intervals violated by an early example) remain plausible, and generalisation from long sets is broader and recency-biased.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/leaky_memory_concept_learning.py`

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 9.0 nats behind sticky_guess_power_law_ranges (2.1× dse)

**Hypothesis:** People do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 as the Erlang-interval model (size principle, 10% noisy examples, a prior weight per rule, an Erlang prior on interval size), but their space of rules also includes "same-step" arithmetic progressions with any starting offset — every class of numbers that are a fixed step apart without being multiples of it (e.g. 2, 7, 12, ... 97 for 82, 67, 72; or 1, 5, 9, ... for 25, 17) — for steps 3 to 12. This family of offset progressions takes a single fitted share of the rule prior, split equally among its members, so examples that share a common spacing are generalised to other numbers on the same step even when no familiar rule (multiples, last digit) fits them.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### category_exemplar_similarity_retrieval — pruned (experiment1 end of experiment)

**Margin:** 9.2 nats behind sticky_guess_power_law_ranges; retired to keep the live set at 8 models: ELPD-LOO rank 21 of 21

**Hypothesis:** People do not work out which concept could have generated the examples. Instead they retrieve familiar number categories stored in memory as sets of remembered members: the familiar rules (odd numbers, powers of 2, multiples of 10, numbers ending in 7, ...) and familiar ranges bounded by multiples of five (the 20s, 1 to 50, 60 to 75, ...). Each category is activated by how similar the examples are, on average, to its stored members, with similarity falling off exponentially with numerical distance at a fitted rate (a generalized-context-model comparison against category exemplars). The target is endorsed in proportion to how similar it is to the nearest member of each activated category, so near-misses partly activate a category and partly earn endorsement (e.g. 31 after 10, 20, 40).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/category_exemplar_similarity_retrieval.py`

### serial_position_weighted_concepts — pruned (experiment1 end of experiment)

**Margin:** 9.7 nats behind sticky_guess_power_law_ranges (2.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same 45 mathematical rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not weigh the examples equally: because the examples are seen in sequence, each example's evidence is weighted by its serial position, with a single fitted order-bias parameter that shifts weight toward the most recent examples (recency) or toward the first ones (primacy) while keeping the total amount of evidence fixed. So the same examples shown in different orders lead to different generalisations, favouring concepts that fit the weighted-up examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_concepts.py`

### lapse_biased_guess_concepts — pruned (experiment1 end of experiment)

**Margin:** 9.7 nats behind sticky_guess_power_law_ranges (2.3× dse)

**Hypothesis:** People do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 as the simplest model (size principle, 10% noisy examples, a prior weight per rule), and on most trials they answer by reporting the posterior probability that the concept contains the target; but on a fitted fraction of trials they lapse — they do not consult the inferred concept at all and instead guess, answering "yes" with a fitted guessing bias. This all-or-none attentional lapse with a biased guess is the one change: it puts a floor and a ceiling on yes rates that are the same for every stimulus, rather than graded softening of confident posteriors.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapse_biased_guess_concepts.py`

### magnitude_compressed_sampling_concepts — pruned (experiment2 end of experiment)

**Margin:** 34.4 nats behind coincidence_triggered_guess_rethink (4.6× dse)

**Hypothesis:** Refinement of power_law_endpoints_erlang_width (same 45 rules and all intervals of 1-100, size principle, 10% noisy examples, a prior weight per rule, power-law endpoint prior and Erlang width prior). The one change is in how people think the examples were sampled: they imagine the program drawing numbers uniformly on a compressed, logarithmic-like mental number line rather than uniformly in raw value, with the degree of compression fitted, so within a range or rule smaller numbers are expected to be drawn more often than larger ones. Because a span among large numbers is then psychologically narrower, the size principle penalises extending a range less for large-number examples than for small-number ones, so generalisation beyond the examples widens in proportion to their magnitude (Weber-like scaling).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/magnitude_compressed_sampling_concepts.py`

### power_law_endpoints_erlang_width — pruned (experiment2 end of experiment)

**Margin:** 34.9 nats behind coincidence_triggered_guess_rethink (4.6× dse)

**Hypothesis:** Refinement of power_law_interval_endpoints: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a prior weight per rule) with interval endpoints whose prior plausibility follows a fitted power law over the number line, but the one change is that they also expect a numerical range to have a characteristic width: the interval prior is additionally weighted by an Erlang density on the interval's size with a fitted typical size (the component taken from erlang_interval_prior_concepts). So where a range lies on the number line and how broad it is both shape the prior, which jointly sets how far people generalise beyond the examples' span for small- versus large-number sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/power_law_endpoints_erlang_width.py`

### round_endpoint_interval_concepts — pruned (experiment2 end of experiment)

**Margin:** 39.2 nats behind coincidence_triggered_guess_rethink (4.9× dse)

**Hypothesis:** Refinement of erlang_interval_prior_concepts: people do the same Bayesian concept learning over the 45 rules and all intervals of 1-100 (size principle, 10% noisy examples, a free prior weight per rule, an Erlang prior on interval size), but when they imagine the program picking a numerical range they favour ranges with round-number boundaries (starting at 1 or a multiple of ten, ending at a multiple of ten, e.g. "20 to 40" or "1 to 50") over arbitrary ones like "17 to 43". The one change is a fitted prior boost for each round endpoint, so generalisation from clustered examples extends to, and then drops off sharply at, the nearest decade boundaries rather than fading smoothly with width alone.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/round_endpoint_interval_concepts.py`

### category_step_exemplar_similarity — pruned (experiment2 end of experiment)

**Margin:** 45.8 nats behind coincidence_triggered_guess_rethink (4.4× dse)

**Hypothesis:** People do not infer a concept; they retrieve familiar number categories stored in memory as ordered lists of their members (odd numbers, powers of 2, numbers ending in 6, ranges bounded by multiples of five, ...), each activated by how similar the examples are to its stored members. They then endorse a target by its exemplar similarity to the nearest example measured along each activated category — how far through the category's member list they must step from an example to reach the target — so a member only a step or two away from an example (32 after 16 in the powers of 2) is endorsed readily while one many steps away (96 after a lone 46 in "ends in 6") is endorsed much less.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/category_step_exemplar_similarity.py`

### partial_strong_sampling_rules_intervals — pruned (experiment1 end of experiment)

**Margin:** 94.8 nats behind sticky_guess_power_law_ranges (4.3× dse)

**Hypothesis:** Refinement of the dropped `tenenbaum_rules_similarity` model (Bayesian concept learning over 33 mathematical rules and all intervals of 1-100, rules sharing a prior fraction equally and intervals weighted by an Erlang density on their size). The one change: learners do not assume every example was deliberately sampled from the concept (pure strong sampling, the size principle); they treat each example as drawn from the concept with probability theta (likelihood 1/|concept|) and otherwise as an arbitrary number that merely happens to fall in it (weak sampling, likelihood independent of concept size), with theta fitted — so the size principle is only partly applied, and generalisation is broader, e.g. beyond the examples' range and to larger rules, than strong sampling predicts. (The original's crash on cross-validation folds, a data-shape dependence on the fold, is also fixed.)

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/partial_strong_sampling_rules_intervals.py`

### representativeness_next_output_judgment — pruned (experiment2 end of experiment)

**Margin:** 196.6 nats behind coincidence_triggered_guess_rethink (9.2× dse)

**Hypothesis:** People do not answer by asking whether the target belongs to the concept they have inferred; they judge by representativeness — how typical the target would be as the program's next output. Using the same concept learning as the sticky single-guess learner (a working guess formed from the first example and kept while later examples fit it), they estimate how much more likely the program is to produce this particular number than an arbitrary number from 1-100, and say "yes" the more strongly that likelihood ratio exceeds a fitted criterion. So a target that belongs only to a broad concept (all even numbers, everything ending in 6) is endorsed much less than one that belongs to a small, specific concept, even when both are certainly members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/representativeness_next_output_judgment.py`

### map_single_concept_selection — pruned (experiment1 end of experiment)

**Margin:** 236.5 nats behind sticky_guess_power_law_ranges (10.2× dse)

**Hypothesis:** People do not average over all the concepts consistent with the examples: they commit to the single most probable concept (a rule such as "powers of 2" or "multiples of 10", or a numeric interval, chosen by a fixed prior and the size principle) and answer "yes" exactly when the target belongs to that one concept, apart from occasional response errors. Generalisation is therefore all-or-none relative to the winning concept, with no graded uncertainty from runner-up concepts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/map_single_concept_selection.py`

### spread_scaled_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 265.3 nats behind sticky_guess_power_law_ranges (9.0× dse)

**Hypothesis:** People do not infer any mathematical rule; they generalize by numerical similarity alone, judging a target more likely to be produced the closer it lies to the individual examples on the number line. The breadth of that similarity gradient is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one, so the same distance counts for less when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/spread_scaled_exemplar_similarity.py`

### exemplar_digit_magnitude_gcm — pruned (experiment1 end of experiment)

**Margin:** 269.8 nats behind sticky_guess_power_law_ranges (10.6× dse)

**Hypothesis:** People do not infer a hidden rule; they judge a target by its summed similarity to the remembered examples (an exemplar, generalized-context-model process). Similarity to each example decays exponentially with a city-block psychological distance built from how far apart the two numbers are in magnitude and whether they differ in their tens digit and in their ones digit, and the probability of "yes" grows with the target's total similarity to the examples relative to a response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_digit_magnitude_gcm.py`

### exemplar_similarity_number_line — pruned (experiment1 end of experiment)

**Margin:** 299.6 nats behind sticky_guess_power_law_ranges (10.8× dse)

**Hypothesis:** People do not induce any rule or interval: they judge a target by its summed perceptual similarity on the number line to the examples shown, with similarity decaying exponentially with numeric distance (an exemplar / GCM account), and say "yes" in proportion to that summed similarity relative to a fixed response criterion. This disagrees most sharply with the rule-and-interval learner in the gaps between widely spaced examples (e.g. 50 for the set 92,14,20,5, where an interval spanning the examples predicts "yes" but similarity predicts "no") and for targets just outside the examples' range (where the interval predicts "no" but similarity predicts "yes").

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_similarity_number_line.py`

### tversky_feature_contrast_similarity — pruned (experiment1 end of experiment)

**Margin:** 310.6 nats behind sticky_guess_power_law_ranges (12.0× dse)

**Hypothesis:** People do not infer a hidden rule or range; they judge a target by Tversky-style feature similarity to the examples. Each number is represented by its salient properties (parity, primality, being a square, cube or power, multiples, final digit, digits it contains, and which band of ten it falls in), each property weighted by how distinctive it is (rarer properties count more), and the target's similarity to the examples is the weighted features it shares with them relative to the features the examples have that it lacks and the features it has that the examples lack (Tversky's ratio contrast model); the probability of "yes" rises with that similarity.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tversky_feature_contrast_similarity.py`

### consistency_attention_similarity — pruned (experiment2 end of experiment)

**Margin:** 443.4 nats behind coincidence_triggered_guess_rethink (13.5× dse)

**Hypothesis:** People judge a target by its similarity to the examples, but they attend to each dimension of number similarity (magnitude band, last digit, parity, divisibility, being a square, power or prime) in proportion to how surprisingly consistently the examples agree on it: a property all examples share counts more the rarer it is among 1-100 and the more examples share it. The target's psychological distance is the summed attention on the shared properties it lacks plus its number-line distance from the nearest example, and the probability of "yes" falls off along this distance — so a far-away number is endorsed when it shares a rare property the examples all have, and a near number is rejected when it breaks one.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/consistency_attention_similarity.py`

### cohesion_selected_metric_similarity — pruned (experiment2 end of experiment)

**Margin:** 450.9 nats behind coincidence_triggered_guess_rethink (12.8× dse)

**Hypothesis:** People judge the target by its similarity to the examples, but on whichever single metric of number similarity the examples themselves cluster on most surprisingly: either closeness on the number line (with a graded, exponentially decaying gradient) or sharing the same remainder when divided by some k from 2 to 10 (same parity, same last digit, both multiples of 7, and so on). They attend to the metric on which the examples are more alike than random numbers would be, and say "yes" in proportion to how similar the target is to the nearest example on that metric — so scattered examples that share a residue (8, 80, 48) are generalised to far numbers with that residue, while examples that share no residue are generalised only to nearby numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/cohesion_selected_metric_similarity.py`

### place_value_grid_block_concepts — pruned (experiment2 end of experiment)

**Margin:** 540.8 nats behind coincidence_triggered_guess_rethink (12.3× dse)

**Hypothesis:** People do not think of the numbers 1-100 as points on a number line but as cells of a place-value grid (tens digit by ones digit), and besides familiar arithmetic rules (even, multiples, powers, squares, primes) the only concepts they entertain are blocks on that grid — a run of tens digits crossed with a run of ones digits, such as "the forties", "ends in 6", "ends in 5 to 7" or "20s and 30s ending in 3 to 7". They learn which block or rule the program uses by Bayesian concept learning with a single sticky working guess formed from the first example, so after 27 and 33 they endorse 23 and 37 (same block of digits) more than 30, which a number-line range would favour.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/place_value_grid_block_concepts.py`

### independent_example_generalization — pruned (experiment2 end of experiment)

**Margin:** 702.5 nats behind coincidence_triggered_guess_rethink (14.5× dse)

**Hypothesis:** People never pool the examples into a single inferred concept. Each example on its own triggers Bayesian generalization (over mathematical rules and numerical ranges, with the size principle applied to that one example), and the target is endorsed according to the average of these separate single-example generalizations. So adding examples does not sharpen generalization onto a shared rule or fill the gap between them as a range: a target is endorsed to the extent that it is something that would be generalized to from the individual examples, one at a time.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment2/model_loop/models/pruned/independent_example_generalization.py`
