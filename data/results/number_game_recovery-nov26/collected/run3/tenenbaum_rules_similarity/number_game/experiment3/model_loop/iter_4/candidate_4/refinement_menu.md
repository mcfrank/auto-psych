# Refinement menu

The models you may refine, other than the incumbent `rule_family_doubt_criterion`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### kind_uncertainty_shifted_criterion — rank 1, 0.5 ± 1.1 nats behind the best (0.5× dse: statistically tied with the best), ELPD-LOO -3663.5

**Hypothesis:** Refinement of `uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are, but the one change is what that uncertainty is about. It is not their spread of belief over individual concepts (which rule, which range width) but their uncertainty about what kind of program they face — a parity rule, a prime rule, a power rule, a multiples rule, a digit rule, or a magnitude range — so examples that leave it open whether the program follows an arithmetic rule or a stretch of the number line move the answer toward "no" (or "yes"), while examples that clearly point to one kind do not, however many concepts of that kind remain open.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/kind_uncertainty_shifted_criterion.py`

### two_cluster_rule_family_doubt — rank 2, 0.6 ± 0.9 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -3663.5

**Hypothesis:** Refinement of `two_cluster_range_concepts`, with one component taken from the incumbent `rule_family_doubt_criterion`. People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), single magnitude ranges and "two separate groups" concepts (two ranges with a gap between them), all judged as a typical share of the 1–100 line and with the size principle. The one change is the yes/no criterion: only doubt about which kind of arithmetic rule the program follows, weighted by how much they believe it is a rule at all, shifts the answer from the inferred membership probability toward "no" (or "yes"), while doubt about whether the numbers form one stretch, two groups or a rule does not — so for widely split sets the two-group concept both carries generalisation away from the empty middle and, by drawing belief away from rules, weakens the criterion shift there.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/two_cluster_rule_family_doubt.py`

### uncertainty_shifted_criterion — rank 3, 0.7 ± 1.9 nats behind the best (0.4× dse: statistically tied with the best), ELPD-LOO -3663.6

**Hypothesis:** People infer the hidden concept exactly as the best model's Bayesian concept learner does, but their yes/no decision criterion depends on how sure they are which concept the program uses: the more their belief is spread over many competing concepts (rules and range widths), the more they shift their answer away from the inferred membership probability toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/uncertainty_shifted_criterion.py`

### first_impression_kind_anchoring — rank 4, 0.9 ± 1.4 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -3663.8

**Hypothesis:** People decide what kind of program they face (a parity, prime, square/cube, power, multiples, final-digit, contained-digit or exception rule, or a stretch of the number line) largely from their first impression of the first example, and later examples only partly adjust that impression (anchoring and adjustment), while which particular concept of the favoured kind it is gets worked out from all the examples. So the same examples in a different order lead to different answers, and a set whose first example suggests a different kind than the whole set does (e.g. 50 then 2, 8, 64) keeps the first-suggested kind in play, giving "yes" to targets that only that kind contains more often than a learner who weighs all examples equally would; otherwise they answer as the best model does, including its shift of criterion with doubt about which kind of rule it is.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/first_impression_kind_anchoring.py`

### beta_share_width_exception_concepts — rank 5, 1.0 ± 1.5 nats behind the best (0.7× dse: statistically tied with the best), ELPD-LOO -3663.9

**Hypothesis:** Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges, with the size principle. The one change is the shape of their prior about how wide a range is: they think of a range as a portion of the bounded 1–100 line and expect it to cover a typical share of it (a beta prior over that share, with its mean and sharpness inferred), rather than holding an unbounded gamma expectation over raw width. So ranges covering nearly the whole line are as implausible to them as very narrow ones, and when the examples are spread widely (e.g. 43, 4, 91, 9) people fall back on a rule instead of an "almost any number" range. As a result they say "yes" less often to numbers that only such a near-total range would include.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/beta_share_width_exception_concepts.py`

### rule_vs_range_doubt_criterion — rank 6, 1.2 ± 1.4 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3664.1

**Hypothesis:** Refinement of `kind_uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are about what kind of program they face. The one change is how coarse that kind judgment is: the only question that moves their criterion is whether the program follows an arithmetic rule at all or picks out a stretch of the number line, so examples that leave rule-versus-range open shift answers toward "no" (or "yes"), while doubt among kinds of rule (say multiples versus final digit) does not shift them, unlike in the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/rule_vs_range_doubt_criterion.py`

### mixed_sampling_uncertainty_criterion — rank 7, 1.3 ± 2.1 nats behind the best (0.6× dse: statistically tied with the best), ELPD-LOO -3664.2

**Hypothesis:** Refinement of `uncertainty_shifted_criterion` with one component taken from `mixed_strong_weak_sampling_concepts`: people infer the hidden concept by the same Bayesian concept learning (family-organised rules with exception rules as their own kind, ranges judged as a typical share of the 1–100 line, occasional stray examples) and shift their yes/no criterion toward "no" (or "yes") the more their belief is spread over competing concepts. The one change is their sampling assumption: they take only an inferred share of the examples as random draws from the concept (the size principle) and the rest as mere members of it, so broad rules and wide ranges stay in play after examples that also fit a narrower concept — which both broadens generalisation to the broad concept's other members and raises the uncertainty that drives the criterion shift for exactly those sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/mixed_sampling_uncertainty_criterion.py`

### relative_concept_size_criterion — rank 8, 1.3 ± 1.2 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3664.3

**Hypothesis:** People infer the hidden concept as the simplest Bayesian concept learner does, but they set their yes/no criterion relative to how many numbers they believe the program produces (an adaptation-level, or "yes-rationing", criterion): when the examples suggest a broad concept that contains much of the 1–100 line, they demand more before saying "yes" to any one target, and when the examples suggest a sparse concept they accept a member more readily — or the reverse, if the inferred dependence is positive. So two targets with the same membership probability are answered differently depending on how large the inferred concept is expected to be.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/relative_concept_size_criterion.py`

### specificity_biased_rule_prior — rank 9, 1.5 ± 1.5 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3664.5

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but within each kind of rule they do not find all rules equally plausible beforehand: a rule's prior plausibility scales with how specific it is (how few numbers it contains), to an inferred degree, so "multiples of 9" is a priori favoured over "multiples of 3" and "multiples of 8" over "multiples of 4" (or disfavoured, if the degree is negative). The sharpest disagreement with the best model lives in example sets that fit a sparse rule nested inside a broader rule of the same kind, especially with only two examples (e.g. 81, 9; 8, 32; 6, 66): there this model confines "yes" to multiples of the larger base and rejects the other multiples of its divisor far more firmly than the size principle alone does.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/specificity_biased_rule_prior.py`

### prototype_magnitude_pull — rank 10, 1.6 ± 1.5 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3664.5

**Hypothesis:** People infer the hidden concept as a Bayesian concept learner does (rules organised by kind with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but they also summarise the examples by a prototype, their average magnitude, and their yes/no answer is pulled by the target's numerical similarity to that prototype. That similarity falls off exponentially with the target's distance on the number line from the examples' mean, with an inferred rate and strength. So targets near the centre of the examples get more "yes" than the inferred concept warrants (or fewer, if the strength is negative), even when no example lies near them (e.g. 50 after 13, 91, or 33 after 2, 64), while targets equally close to a single example but far from the centre get no such pull.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/prototype_magnitude_pull.py`

### range_misperception_uncertainty_criterion — rank 11, 1.6 ± 2.0 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -3664.5

**Hypothesis:** Refinement of `range_size_misperception_concepts`, with one component taken from `uncertainty_shifted_criterion`. People do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, and they misjudge how many numbers a range holds (its felt size grows as an inferred power of its true width), so wide ranges are penalised too little or too much by the size principle. The one change is the yes/no criterion: the more their belief is spread over competing concepts, the further people shift away from the inferred membership probability, toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". Because a compressed sense of range size keeps many range widths open for three- and four-example sets, this shift is strongest for exactly the sets where magnitude generalisation already reaches beyond the examples' span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/range_misperception_uncertainty_criterion.py`

### mixed_sampling_kind_uncertainty — rank 12, 1.7 ± 1.6 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3664.6

**Hypothesis:** Refinement of `kind_uncertainty_shifted_criterion` with one component taken from `mixed_strong_weak_sampling_concepts`: people infer the hidden concept by the same Bayesian concept learning (family-organised rules with exception rules as their own kind, ranges judged as a typical share of the 1–100 line, occasional stray examples) and shift their yes/no criterion with how unsure they are what kind of program they face (which family of rule, or a magnitude range). The one change is their sampling assumption: they take only an inferred share of the examples as random draws from the concept (the size principle) and the rest as mere members of it, so broad rules and wide ranges of other kinds stay in play after examples that also fit a narrow concept — broadening generalisation to the broad concept's members and raising exactly the kind-level doubt that moves the criterion for those sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/mixed_sampling_kind_uncertainty.py`

### person_specific_response_bias — rank 13, 1.7 ± 1.2 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -3664.6

**Hypothesis:** People infer how likely the target is to belong to the hidden concept as the simplest Bayesian concept learner does (rules with a family-organised prior and exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but each person brings a stable bias toward answering "yes" or "no" that holds across their whole session: their decision criterion is shifted, by a person-specific amount drawn from a population distribution, so the log-odds of saying "yes" is the inferred membership log-odds plus that person's own bias. So, for the same examples and target, a yes-prone person accepts more targets of every kind (including clear non-members), and a no-prone person rejects more (including shown examples), rather than everyone answering with the same probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/person_specific_response_bias.py`

### two_cluster_kind_doubt_criterion — rank 14, 1.7 ± 1.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3664.6

**Hypothesis:** Refinement of `two_cluster_range_concepts`, with one component taken from the incumbent `kind_uncertainty_shifted_criterion`. People do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), single magnitude ranges, and "two separate groups" concepts (two ranges with a gap between them), all judged as a typical share of the 1–100 line and with the size principle. The one change is the yes/no criterion: the more the examples leave open what kind of program it is, including whether the numbers form one stretch of the line, two separate groups or an arithmetic rule, the further people shift their answer from the inferred membership probability toward "no" (or toward "yes" if the inferred shift is negative). So widely split sets such as 13, 91, which leave one range, two clusters and a rule all in play, get a shifted criterion that the single-range incumbent cannot produce.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/two_cluster_kind_doubt_criterion.py`

### doubt_driven_magnitude_fallback — rank 15, 1.7 ± 1.7 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -3664.7

**Hypothesis:** Refinement of the incumbent `rule_family_doubt_criterion`: people infer the hidden concept as its Bayesian concept learner does, and their doubt about which kind of arithmetic rule the program follows (parity, primes, powers, multiples, final digit, ...), weighted by how much they believe it is a rule at all, shapes their answer. The one change is what that doubt does: rather than shifting their yes/no criterion toward "no" or "yes", it makes them fall back on the number line, so the more unsure they are which kind of rule it is, the more (to an inferred degree) they judge the target by how a stretch of the number line around the examples would generalise. Under rule-kind doubt, numbers near the examples therefore gain "yes" answers and members of only one candidate rule lose them, while sets that clearly point to one kind of rule are judged by the full posterior.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/doubt_driven_magnitude_fallback.py`

### single_kind_commitment_inference — rank 16, 1.9 ± 1.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3664.8

**Hypothesis:** People reason about the hidden program in two steps, as in single-category inference (Murphy & Ross): they first work out what kind of program it is — a parity rule, a prime rule, a square/cube rule, a power rule, a multiples rule, a final-digit rule, a contained-digit rule, an exception rule, or a magnitude range — and then judge the target mostly within the kind they favour, rather than weighing every kind by its full posterior plausibility. So when the examples leave two kinds open (e.g. "multiples of 4" versus "numbers between 20 and 60"), the less favoured kind is discounted to an inferred degree, and targets that only it would contain get fewer "yes" answers than a fully Bayesian learner predicts (or more, if people hedge across kinds more evenly than the posterior warrants).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/single_kind_commitment_inference.py`

### two_cluster_range_concepts — rank 17, 1.9 ± 1.8 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3664.8

**Hypothesis:** People do the same Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but their magnitude concepts are not limited to one unbroken stretch of the line: when the examples fall into two separated groups, they can take the program to produce "small numbers and large numbers" — two separate ranges with a gap between them — as a kind of concept with its own inferred plausibility. The sharpest disagreement with the best model lies in targets in the empty middle of widely split sets (e.g. 50 after 13, 91 or 43, 4, 91, 9; 60 after 20, 32, 92): this model says "no" to them while still saying "yes" to numbers near either group, whereas a single-range learner must either cover the gap or fall back on a rule.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/two_cluster_range_concepts.py`

### range_size_misperception_concepts — rank 18, 2.0 ± 1.6 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3664.9

**Hypothesis:** People do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they misjudge how many numbers a range holds: a range's felt size grows as an inferred power of its true width, so (if the power is below one) wide ranges seem to hold fewer numbers than they do and are penalised too little by the size principle — or, above one, too much. Because this distortion acts on the evidence each example gives rather than on the prior, it shows up most with more examples: with a compressed sense of range size, adding examples inside a span narrows magnitude generalisation less than the normative learner predicts, so "yes" extends further beyond the examples' span for three- and four-example sets.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/range_size_misperception_concepts.py`

### mixed_strong_weak_sampling_concepts — rank 19, 2.1 ± 1.8 nats behind the best (1.1× dse: statistically tied with the best), ELPD-LOO -3665.0

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples. The one change is their assumption about how the examples were generated: they believe only some genuine examples (an inferred share) were drawn at random from the concept, which is what drives the size principle, while the rest were merely numbers the concept happens to contain, chosen without regard to how big the concept is (mixed strong/weak sampling). So the size principle is diluted most for big concepts, and after examples that fit both a narrow and a broad concept (multiples of 9 vs multiples of 3, a tight range vs a wide one) people keep the broad concept in play and say "yes" to its other members more than pure random sampling predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/mixed_strong_weak_sampling_concepts.py`

### range_first_satisficing_search — rank 20, 2.2 ± 1.4 nats behind the best (1.5× dse: statistically tied with the best), ELPD-LOO -3665.1

**Hypothesis:** People search for the hidden program in a fixed order of ease: they first ask whether the examples simply fall in some stretch of the number line, and they go on to look for an arithmetic rule (parity, multiples, powers, digits and so on) less thoroughly the better such a stretch already accounts for the examples (satisficing), to an inferred degree — or more thoroughly, if that degree is negative. Otherwise they infer the concept as a Bayesian concept learner does, so for tightly clustered examples rule members outside the cluster get fewer "yes" answers than full search predicts, while for widely scattered examples rules keep their full weight.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/range_first_satisficing_search.py`

### kind_disagreement_target_criterion — rank 21, 2.5 ± 2.0 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -3665.4

**Hypothesis:** Refinement of `kind_uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion when their doubt comes from not knowing what kind of program they face. The one change is that this doubt is judged for the target in question rather than for the example set as a whole: the criterion moves toward "no" (or "yes") in proportion to how much of their uncertainty about this particular target stems from the open kinds disagreeing about it (e.g. a range would include it but the favoured rule would not), as opposed to fuzziness within one kind (where a range's edge lies, which multiples rule) — so on the same set, a target that the remaining kinds agree on is answered by the membership probability alone, while one they disagree on is shifted.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/kind_disagreement_target_criterion.py`

### mixed_sampling_prior_leaking_judgment — rank 22, 3.1 ± 1.9 nats behind the best (1.6× dse: statistically tied with the best), ELPD-LOO -3666.1

**Hypothesis:** Refinement of `mixed_strong_weak_sampling_concepts` with one component taken from `beta_share_width_prior_leaking_judgment`: people infer the hidden concept exactly as the incumbent's learner does (family-organised rules with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, occasional stray examples, and only an inferred share of examples taken as random draws that drive the size principle). The one change is in the yes/no judgment: it is pulled by how plausible the target was before any examples were shown, so the log-odds of saying "yes" is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership. A priori typical numbers (mid-range numbers that most moderately wide ranges contain, members of many rules) therefore get more "yes" than the examples warrant, or less if the share is negative, on top of the broad-concept generalisation that mixed sampling produces.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/mixed_sampling_prior_leaking_judgment.py`

### sample_size_shifted_criterion — rank 23, 3.3 ± 1.9 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -3666.2

**Hypothesis:** People infer the hidden concept as a Bayesian concept learner does (rules organised by kind with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but how readily they act on that inference depends on how much evidence they were shown: their yes/no criterion shifts by an inferred amount with each additional example (a sample-size heuristic). So for the same membership probability, people answer "yes" differently after one example than after four, regardless of how sure the examples make them about which concept, or which kind of concept, the program uses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/sample_size_shifted_criterion.py`

### boundary_examples_anchor_ranges — rank 24, 4.0 ± 1.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3667.0

**Hypothesis:** When people consider whether the examples come from a magnitude range, they judge the range mainly by the smallest and largest examples, which mark its edges. The examples in between tell them nothing new about where the edges are, so they count for less (to an inferred degree) as evidence that the range is narrow, while every example counts fully when they test an arithmetic rule. As a result, with three or four examples people keep entertaining wider ranges, and a range explanation stays more competitive with a rule, than an ideal learner who weighs every example equally would.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/boundary_examples_anchor_ranges.py`

### digit_parity_family_doubt — rank 25, 4.9 ± 2.9 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -3667.8

**Hypothesis:** Refinement of `rule_family_doubt_criterion`, with one component taken from `digit_parity_concepts`. People infer the hidden concept as the incumbent's Bayesian learner does and shift their yes/no criterion with how unsure they are which kind of arithmetic rule the program follows (weighted by how much they believe it is a rule at all); the one change is that their repertoire of rule kinds also includes a digit-based kind — "numbers written only with odd digits" and "numbers written only with even digits" — with its own inferred plausibility. So after examples such as 31, 3, 1, 15 people say "yes" to 71 or 95 but "no" to 63 or 85, and sets that leave this digit kind open alongside parity or other rules carry more kind-of-rule doubt, shifting the criterion where the incumbent's does not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/digit_parity_family_doubt.py`

### digit_parity_concepts — rank 26, 6.1 ± 3.1 nats behind the best (1.9× dse: statistically tied with the best), ELPD-LOO -3669.0

**Hypothesis:** People do the same Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they also read numbers by their written digits and hold a kind of concept the current models lack: "numbers written only with odd digits" (1, 3, 5, ..., 11, 13, ..., 99) and "numbers written only with even digits" (2, 4, 6, 8, 20, 22, ..., 88), with this kind's plausibility inferred. So after examples such as 31, 3, 1, 15 or 9, 33, 73, 99 people say "yes" to odd numbers like 71, 77 or 95 but "no" to odd numbers with an even tens digit such as 63 or 85, rather than spreading "yes" evenly over all odd numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/digit_parity_concepts.py`

### best_explanation_contrast — rank 27, 274.5 ± 29.2 nats behind the best (9.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3937.4

**Hypothesis:** People do not weigh every concept that could explain the examples; they answer by comparing two best explanations: the single most plausible concept (rule or magnitude range, judged with the same priors and size principle as the Bayesian learner) that includes the target, against the single most plausible concept that excludes it. They say "yes" with a probability that grows with how much better the best target-including explanation is than the best target-excluding one, so a target is accepted when one good concept contains it, regardless of how many other good concepts do or do not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/best_explanation_contrast.py`

### individual_rule_vs_range_tendency — rank 28, 324.4 ± 34.1 nats behind the best (9.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3987.3

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is that people differ, stably across their session, in how readily they take the hidden program to follow an arithmetic rule rather than a magnitude range: each person has their own prior rule-versus-range odds, drawn from a population distribution whose mean and spread are inferred. For the same examples, a "rule-minded" person confines "yes" to members of the fitting rule while a "magnitude-minded" person spreads "yes" over nearby numbers, so their answers to the same (set, target) pair disagree systematically rather than at random.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/individual_rule_vs_range_tendency.py`

### conjunctive_shared_feature_exemplars — rank 29, 901.9 ± 44.4 nats behind the best (20.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -4564.9

**Hypothesis:** People do not infer a hidden concept; they compare the target with each stored example using a shared-feature (additive-clustering) similarity: two numbers are similar to the extent that they share familiar number properties (parity, multiples, powers, squares, primes, final digit, digits contained), each property weighted by how salient its kind is and how distinctive it is (properties held by few numbers count more), plus their closeness in magnitude. The judgment is conjunctive: the target is accepted to the extent that it is similar to every example (the geometric mean of its normalised similarities to the examples), so a target must resemble all the examples, not just one, to be called "yes".

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/conjunctive_shared_feature_exemplars.py`

## Pruned models (out of the set; narrowest margin first)

### beta_share_width_prior_leaking_judgment — pruned (experiment2 end of experiment)

**Margin:** 2.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `prior_leaking_membership_judgment`: people infer the hidden concept exactly as the incumbent's Bayesian learner does (family-organised rules with exception rules as their own kind, magnitude ranges judged as a typical share of the 1–100 line, the size principle), but their yes/no judgment is pulled by how plausible the target was before any examples were shown. The log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership (under the same share-of-the-line range prior), so a priori typical numbers — mid-range numbers that most moderately wide ranges contain, and numbers belonging to many rules — get more "yes" than the examples alone warrant (or less, if the share is negative).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/beta_share_width_prior_leaking_judgment.py`

### beta_share_parity_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 3.0 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts` with one component taken from `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle, but they can also entertain the conjunctive concept "odd (or even) numbers within a range" (e.g. "even numbers above 60") as its own kind of concept, with its own inferred plausibility and the same share-of-the-line width prior as plain ranges. So after 4, 66, 78, 96 people say "yes" to large even numbers (70, 74) but "no" to small ones and to odd numbers between the examples, and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" over every number in the range or over all even (odd) numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/beta_share_parity_range_concepts.py`

### inferred_sampling_regime_concepts — pruned (experiment2 end of experiment)

**Margin:** 3.2 nats behind mixed_strong_weak_sampling_concepts (2.6× dse)

**Hypothesis:** Refinement of `mixed_strong_weak_sampling_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples, and they allow both random sampling from the concept (the size principle) and weak sampling (examples merely members of it). The one change is where that uncertainty sits: rather than each example independently being a random draw or not, people are unsure whether the program as a whole draws its outputs at random from its concept, and they infer this from the whole set together with the concept. So a suspicious coincidence (every example fits a small rule, e.g. 2, 8, 16, 64) is itself taken as evidence of random sampling and sharpens generalisation onto the small rule, while sets that no small concept explains well tilt belief towards weak sampling and keep broad rules and wide ranges in play.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/inferred_sampling_regime_concepts.py`

### range_placement_prior_concepts — pruned (experiment2 end of experiment)

**Margin:** 3.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 12 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges, with the size principle and the same expectation that a range covers a typical share of the 1–100 line. The one change is that people also expect ranges to sit in a typical place on the line: instead of every placement of a range of a given width being equally plausible, they hold a prior (inferred mean position and sharpness) over where a range's centre lies, so ranges among, say, small numbers can be a priori more plausible than equally wide ranges among large numbers. As a result, examples falling in a disfavoured region of the line are explained by a rule more readily, and "yes" to nearby numbers there is lower, than for the same pattern of examples in a favoured region.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_placement_prior_concepts.py`

### decade_kind_beta_share_concepts — pruned (experiment2 end of experiment)

**Margin:** 3.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 15

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`, with one component taken from the pruned `decade_concept_family_prior`. People do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. They also hold the decade as a ready-made concept ("single-digit numbers", "numbers in the teens", "numbers in the 60s"), which forms its own kind of rule with its own inferred plausibility. So when the examples share a tens digit (e.g. 63, 67 or 2, 8), people confine "yes" sharply to that decade, accepting 60 and 69 but rejecting 70 and 59. A free-floating range of the same width would give a smooth fall-off around the examples instead.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_kind_beta_share_concepts.py`

### arbitrary_exception_rule_prior — pruned (experiment1 end of experiment)

**Margin:** 3.8 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 10

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but a rule with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") is not just another power rule to them — it belongs to its own kind, with its own inferred, presumably very low, plausibility. The one change is giving these exception-laden rules their own family in the rule prior, so after seeing powers of 2 people firmly reject 37 (and accept 32), instead of giving 37 the roughly one-in-three "yes" that the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/arbitrary_exception_rule_prior.py`

### decade_concept_family_prior — pruned (experiment1 end of experiment)

**Margin:** 4.3 nats behind gamma_width_parity_range_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 10

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior` (the incumbent): people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, rule prior shared by kind of rule, exception-laden rules as their own kind), but they also hold a ready-made kind of concept that the incumbent lacks: the decade, "numbers in the 60s" (same tens digit), as a distinct kind of rule with its own inferred plausibility. The one change is adding the ten decade concepts as a new family in the rule prior, so when the examples share a tens digit (e.g. 61, 64, 67) people confine "yes" sharply to that decade, more firmly than an arbitrary range of the same width.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_concept_family_prior.py`

### prior_leaking_membership_judgment — pruned (experiment2 end of experiment)

**Margin:** 5.1 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 13 of 15

**Hypothesis:** People infer the hidden concept as a normative Bayesian concept learner does (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a preferred width, the size principle), but their yes/no judgment is distorted by how plausible the target was before they saw any examples: the log-odds of saying yes is the posterior log-odds that the concept contains the target, shifted by an inferred share of the target's prior log-odds of membership. So a priori "typical" numbers (mid-range, even, multiples) get more "yes" than the examples warrant (prior leakage), or, if the share is negative, less (contrast against what was expected anyway).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/prior_leaking_membership_judgment.py`

### dominated_rule_discount_concepts — pruned (experiment2 end of experiment)

**Margin:** 5.1 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is which rules they seriously entertain given the examples: when every example also fits a strictly narrower rule (all multiples of 9 fit "multiples of 3"; 4, 16, 64 fit both "powers of 4" and "powers of 2"), they settle on the narrower rule and discount the broader, dominated one by an inferred factor beyond what the size principle already does (arbitrary-exception rules never count as the narrower rule). So after 36, 45, 54, 81 people say "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9 more sharply than the incumbent predicts — unlike a prior favouring specific rules, the discount applies only when a narrower rule actually fits the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/dominated_rule_discount_concepts.py`

### beta_share_without_replacement_concepts — pruned (experiment2 end of experiment)

**Margin:** 6.4 nats behind mixed_strong_weak_sampling_concepts (2.3× dse)

**Hypothesis:** Refinement of `without_replacement_sampling_concepts` with one component taken from `beta_share_width_exception_concepts`: people assume the program never repeats a number, so they treat the examples as drawn from the hidden concept without replacement (each further example is one of the concept's members not yet shown), and they do Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges on that basis. The one change is their prior about how wide a range is: they judge a range as a typical share of the bounded 1–100 line, so near-total "almost any number" ranges are implausible. As a result, when the examples use up a good part of a small rule (e.g. 36, 45, 54, 81 out of the multiples of 9), people favour that most specific rule over the broader one around it (multiples of 3) more strongly than random sampling with replacement allows, saying "yes" to other multiples of 9 and "no" to multiples of 3 that are not multiples of 9.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/beta_share_without_replacement_concepts.py`

### gamma_width_exception_family_concepts — pruned (experiment2 end of experiment)

**Margin:** 7.2 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 14 of 15

**Hypothesis:** Refinement of `gamma_width_family_prior_concepts`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, family-organised rule prior, and a gamma prior over range width whose sharpness is inferred along with its typical width). The one change, taken from `arbitrary_exception_rule_prior`, is that rules with an arbitrary exception ("powers of 2 except 32", "powers of 2 plus 37") form their own kind of rule with their own, presumably very low, plausibility, instead of being ordinary power rules. So people expect ranges of a sharply characteristic width and also firmly reject 37 (and accept 32) after seeing powers of 2.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gamma_width_exception_family_concepts.py`

### inferred_stray_rate_concepts — pruned (experiment2 end of experiment)

**Margin:** 7.2 nats behind mixed_strong_weak_sampling_concepts (2.1× dse)

**Hypothesis:** Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges with a gamma width preference and the size principle, and they also allow that any example may be a stray number the program produced at random rather than a member of the concept. The one change is that how often they expect such strays is no longer fixed at one example in ten but is a belief inferred from their judgments: the more strays they expect, the more readily they keep concepts that miss an example (and so sometimes reject an example they were shown), and the weaker the size principle becomes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/inferred_stray_rate_concepts.py`

### hundred_chart_block_concepts — pruned (experiment2 end of experiment)

**Margin:** 7.7 nats behind mixed_strong_weak_sampling_concepts; retired to keep the live set at 8 models: ELPD-LOO rank 15 of 15

**Hypothesis:** People picture the numbers 1–100 laid out on the familiar hundred chart (tens digit as the row, units digit as the column), and besides arithmetic rules and ranges on the number line they entertain rectangular blocks of that chart as concepts, such as "numbers in the 20s to 80s that end in 2 to 6", with this kind's plausibility inferred and its size judged as a share of the chart. With the size principle, examples that cluster in a few columns across several rows (e.g. 22, 62, 26, 82) lead people to say "yes" to other numbers in that block (42, 66) and "no" to numbers outside its columns (28, 63), instead of spreading "yes" over all even numbers or one wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/hundred_chart_block_concepts.py`

### first_example_trust_concepts — pruned (experiment2 end of experiment)

**Margin:** 8.1 nats behind mixed_strong_weak_sampling_concepts (2.2× dse)

**Hypothesis:** People treat the first example they see differently from the later ones when judging how far each example is a genuine output of the hidden concept: the first example sets the concepts they entertain and is trusted to belong to the concept at its own inferred rate. Each later example is trusted at the usual fixed rate, and one that does not fit can be dismissed as a stray output. Apart from this, they generalise by the same Bayesian concept learning as the best model (size principle, rule prior organised by kind with exception rules as their own kind, gamma prior over range width). So if the first example is trusted more, an odd-one-out example is discounted when it comes late in the set (8, 24, 48, 63 still suggests multiples of 8) but forces a broader concept when it comes first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_example_trust_concepts.py`

### gamma_width_parity_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 8.9 nats behind mixed_strong_weak_sampling_concepts (2.2× dse)

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the size principle. The one change, taken from `gamma_width_exception_family_concepts`, is the shape of their expectation about how wide a range (plain or parity-restricted) is: a gamma prior whose sharpness is inferred along with its typical width, instead of the fixed, loose Erlang form — so people may expect ranges of one sharply characteristic width, strongly disfavouring both very narrow and very wide ranges around the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gamma_width_parity_range_concepts.py`

### gamma_width_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 10.1 nats behind gamma_width_parity_range_concepts (2.2× dse)

**Hypothesis:** Refinement of `erlang_interval_family_prior_concepts` (the incumbent): people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with the size principle. The one change is the shape of their prior expectation about how wide a range is: instead of a fixed Erlang form (whose preference for a typical width is always of the same, fairly loose shape), the width prior is a gamma distribution whose shape is inferred along with its typical width, so people may hold a sharp expectation that ranges have one characteristic width (strongly disfavouring both narrow and very wide ranges) or, at the other extreme, a monotone preference for the narrowest range that fits the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/gamma_width_family_prior_concepts.py`

### erlang_interval_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 11.0 nats behind gamma_width_parity_range_concepts (2.4× dse)

**Hypothesis:** Refinement of `erlang_interval_size_concept_learning` (the narrowest-losing live model): people do Bayesian concept learning over the same 45 rules and magnitude ranges, with the size principle and an Erlang prior that favours ranges of a characteristic, inferred width over very wide or single-number ones. The one change is how prior belief is shared among the rules: instead of an independent weight for each of the 45 rules, people first favour a kind of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each kind with its own inferred plausibility, and treat rules of the same kind as equally likely. So whole kinds of rule (e.g. "contains the digit d") can be implausible a priori, while magnitude generalisation follows ranges of a preferred size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_family_prior_concepts.py`

### times_table_familiarity_concepts — pruned (experiment2 end of experiment)

**Margin:** 12.3 nats behind mixed_strong_weak_sampling_concepts (2.6× dse)

**Hypothesis:** Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with the size principle. The one change is how prior belief is shared among the "multiples of m" rules: not equally, but by how familiar each times table is to them, with a separate inferred plausibility for each table that is not tied to how many numbers it contains. So familiar tables (e.g. multiples of 5 or 9) are a readier explanation of examples they fit than unfamiliar (7, 11, 12) or broader (3) tables, and after examples that are all multiples of 9 people confine "yes" to the 9-table more firmly than the size principle alone implies (or, if the 3-table is the more familiar, spread it further).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/times_table_familiarity_concepts.py`

### rule_family_prior_concepts — pruned (experiment1 end of experiment)

**Margin:** 12.6 nats behind gamma_width_parity_range_concepts (2.6× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals (size principle, 10% of examples random), but their prior over rules is organised by kind of rule rather than flat: they first favour a family of rule (parity, primes, squares/cubes, powers, multiples, last digit, contained digit), each with its own inferred plausibility, and rules within a family are equally likely. The one change is replacing the mixture's uniform rule prior with this inferred family-level prior, so whole kinds of rule (e.g. "contains the digit d") can be a priori implausible while others (multiples, last digit) dominate generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/rule_family_prior_concepts.py`

### iter2_candidate3 — pruned (experiment1 end of experiment)

**Margin:** 12.7 nats behind gamma_width_parity_range_concepts (2.8× dse)

**Hypothesis:** Refinement of the incumbent `erlang_interval_family_prior_concepts`: people do the same Bayesian concept learning (45 rules with a family-organised prior, magnitude ranges with an Erlang width prior, size principle), but on some judgments they do not consult the inferred concept at all and simply guess, answering "yes" with their own guessing bias. The one change is this attentional lapse: on an inferred share of trials the answer is a biased guess, so targets the learner is certain about are still sometimes answered the other way — "yes" to numbers outside every plausible concept and "no" to numbers inside it — rather than with near-zero or near-one probability.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/iter2_candidate3.py`

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Margin:** 14.2 nats behind gamma_width_parity_range_concepts (2.9× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude ranges as the best model, but they update sequentially as the examples are read and do not weigh every example equally: each example's evidence is weighted by its position in the sequence, decaying (or growing) geometrically with how far it lies before the last example shown, at an inferred rate. A recency rate means earlier examples are partly forgotten, so longer sets narrow generalisation less than the size principle predicts; a primacy rate means early examples are overcounted, so longer sets narrow it more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_evidence.py`

### lapse_guess_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 14.4 nats behind gamma_width_parity_range_concepts (3.0× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept by Bayesian concept learning over rules (with a family-organised rule prior) and magnitude ranges, using the size principle, and normally answer with that probability; but on some fraction of judgments they lapse (inattention, a slip, not engaging with the examples) and answer by a fixed guessing tendency instead, saying "yes" with an inferred guessing rate regardless of the examples. So even targets the concept clearly excludes get a floor of "yes" answers, and targets it clearly includes get a ceiling below certainty, both set by one lapse rate and one guessing bias.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapse_guess_concept_learning.py`

### evidence_overweighting_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 14.9 nats behind gamma_width_parity_range_concepts (3.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules (with a family-organised rule prior) and magnitude intervals as the best model, using the size principle, but they mis-weight the evidence: each example's likelihood is raised to an inferred power, so the examples count for more (or less) than a normative learner would let them. If people over-weight evidence, their generalisation collapses onto the smallest fitting concept faster as examples accumulate than the size principle predicts, which would explain why their yes-rate drops more from two- to four-example sets than the best model expects.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/evidence_overweighting_concept_learning.py`

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 14.9 nats behind gamma_width_parity_range_concepts (2.3× dse)

**Hypothesis:** Refinement of `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules and magnitude ranges (size principle, Erlang prior over range width, prior shared by kind of rule), but their space of rules also contains arithmetic progressions that do not start at zero — "every third number starting from 1", "odd multiples of 3" (3, 9, 15, ...), "every fourth number starting from 2" — i.e. all numbers leaving the same nonzero remainder under a small step of 3 to 6. These offset progressions form their own kind of rule with its own inferred, presumably low, plausibility, so after examples such as 69, 9, 39, 21 people can generalise only to the odd multiples of 3 and reject other odd numbers such as 19.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### without_replacement_sampling_concepts — pruned (experiment2 end of experiment)

**Margin:** 15.1 nats behind mixed_strong_weak_sampling_concepts (3.0× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior, arbitrary-exception rules as their own kind, magnitude ranges of a characteristic width), but they assume the program never repeats a number, so they treat the examples as drawn from the concept without replacement: each further example is one of the concept's members not yet shown. As a result, examples that nearly exhaust a small concept (e.g. 8, 64 or 81, 9) point to that concept, such as cubes or powers of 3, much more strongly than under sampling with replacement, and generalisation narrows faster onto small, nearly exhausted rules.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/without_replacement_sampling_concepts.py`

### gamma_width_doubling_chain_concepts — pruned (experiment2 end of experiment)

**Margin:** 19.3 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `doubling_chain_concepts`: people do Bayesian concept learning over arithmetic rules (organised by kind, with exception rules as their own kind), magnitude ranges and odd/even-within-a-range concepts, using the size principle, and they also entertain multiplicative chains — each number double (or triple) the one before, starting from a number other than 1 (e.g. 5, 10, 20, 40, 80) — as a kind of rule with its own plausibility. The one change, taken from the best model `gamma_width_exception_family_concepts`, is their expectation about how wide a range is: a gamma prior whose sharpness is inferred along with its typical width, in place of the fixed, loose Erlang form. With ranges judged as the best model judges them, chains are put to a fair test: after a chain-like set such as 5, 10, 20 this model still predicts "yes" to 40 and 80 and "no" to other multiples of 5 such as 15, 25 and 30, where the best model spreads "yes" over all multiples of 5.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gamma_width_doubling_chain_concepts.py`

### pedagogical_teacher_sampling_concepts — pruned (experiment2 end of experiment)

**Margin:** 19.9 nats behind mixed_strong_weak_sampling_concepts (3.2× dse)

**Hypothesis:** People assume the examples were chosen by a helpful teacher rather than drawn at random from the concept: they reason that a program meaning a given concept would pick examples that point distinctively to that concept (numbers that few other plausible concepts would also produce), and they infer the concept, over the same rules and magnitude ranges as the best model, by asking which concept such a teacher would most likely have chosen these examples for. So when every example also fits a narrower rule (e.g. all multiples of 9), people conclude the teacher meant the narrower rule — a broader one (multiples of 3) would have been shown with examples that are not multiples of 9 — and they reject the broader rule's other members more firmly than random sampling alone implies.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/pedagogical_teacher_sampling_concepts.py`

### example_recall_limited_concepts — pruned (experiment2 end of experiment)

**Margin:** 20.5 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, size principle), but their memory for the examples is limited: each example shown is retained, independently, only with an inferred probability, and they generalise from whichever examples they still hold. So a judgment is sometimes based on only part of the set, which broadens generalisation for longer sets (a forgotten example no longer rules out a broader or different concept) in proportion to how many examples could have been lost.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/example_recall_limited_concepts.py`

### parity_within_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 20.7 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of the incumbent `arbitrary_exception_rule_prior`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges with a preferred width and the size principle, but they can also entertain a conjunctive kind of concept, "odd (or even) numbers within a range" (e.g. "even numbers above 60"), with its own inferred plausibility and the same width preference. The one change is adding this parity-within-a-range kind of concept, so that after 4, 66, 78, 96 people say yes to large even numbers (70, 74) but no to small ones (8, 22), and after 23, 35, 95, 83 they reject even numbers between the examples, rather than spreading "yes" evenly over all even (or all in-range) numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/parity_within_range_concepts.py`

### decade_and_parity_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 22.8 nats behind mixed_strong_weak_sampling_concepts (2.8× dse)

**Hypothesis:** Refinement of `decade_concept_family_prior` (tied with the best): people do the same Bayesian concept learning over rules (family-organised prior, exception-laden rules as their own kind, and decades such as "numbers in the 60s" as their own kind of rule) and magnitude ranges with a preferred width and the size principle. The one change, taken from `parity_within_range_concepts`, is that people can also entertain "odd (or even) numbers within a range" as a kind of concept with its own inferred plausibility, so examples sharing a tens digit (61, 64, 67) confine "yes" to that decade while examples such as 4, 66, 78, 96 confine it to large even numbers — two ready-made ways of bounding generalisation that the incumbent and the decade model each have only one of.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/decade_and_parity_range_concepts.py`

### rule_noticing_grows_with_examples — pruned (experiment2 end of experiment)

**Margin:** 23.3 nats behind mixed_strong_weak_sampling_concepts (3.0× dse)

**Hypothesis:** People do the same Bayesian concept learning over arithmetic rules, magnitude ranges and odd/even-within-a-range concepts as the best model, but whether a rule even comes to mind depends on how many examples they have seen: the prior odds of thinking of a rule rather than a magnitude range grow steadily, at an inferred rate, with each additional example. So after one or two examples people generalise mostly by magnitude, and after four examples they look for a rule more readily than the examples' likelihood alone would warrant.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/rule_noticing_grows_with_examples.py`

### log_number_line_range_concepts — pruned (experiment1 end of experiment)

**Margin:** 23.7 nats behind gamma_width_parity_range_concepts (4.1× dse)

**Hypothesis:** Refinement of `rule_family_prior_concepts`: people do the same Bayesian concept learning over the 45 rules (with the family-organised rule prior) and magnitude ranges of 1-100, with size-principle sampling, but they picture the ranges on a compressed, roughly logarithmic mental number line rather than a linear one. The one change is the prior over ranges: a range's endpoints are equally likely to fall anywhere on that mental line, with the degree of compression inferred, so ranges among small numbers (where numbers are spread out on the mental line) are a priori more plausible than equally long ranges among large numbers (where they are crowded together). As a result, examples among large numbers are explained by a range less readily and by a rule more often than the incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_number_line_range_concepts.py`

### multiples_within_range_concepts — pruned (experiment2 end of experiment)

**Margin:** 23.7 nats behind mixed_strong_weak_sampling_concepts (3.1× dse)

**Hypothesis:** Refinement of the incumbent `parity_within_range_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind), magnitude ranges and "odd (or even) numbers within a range", with the Erlang width preference and the size principle. The one change is that the conjunctive "rule within a range" idea extends beyond parity: people also entertain "multiples of m within a range" (e.g. "multiples of 3 between 10 and 60", "multiples of 9 from 27 to 81") as a further kind of concept with its own inferred plausibility and the same width preference. So after 39, 60, 12, 45 or 63, 27, 81 they accept multiples near the examples (e.g. 36, 54) but reject multiples far outside the examples' span (e.g. 90, 99, or 3), instead of spreading "yes" over all multiples of 3 or over every number in the range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/multiples_within_range_concepts.py`

### doubling_chain_concepts — pruned (experiment2 end of experiment)

**Margin:** 28.8 nats behind mixed_strong_weak_sampling_concepts (3.2× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (family-organised rule prior with exception rules as their own kind, magnitude ranges and odd/even-within-a-range concepts with a preferred width, and the size principle), but they also hold a kind of concept the best model lacks: multiplicative chains, in which each number is double (or triple) the one before, starting from some number other than 1 (e.g. 3, 6, 12, 24, 48, 96 or 5, 10, 20, 40, 80 or 2, 6, 18, 54). This kind has its own inferred plausibility. The disagreement with the best model shows up on example sets that form such a chain (e.g. 5, 10, 20 or 6, 12, 24): there this model predicts "yes" to the next links (40, 80 or 48, 96) and "no" to other multiples (15, 25, 30 or 18, 30, 36), whereas the best model spreads "yes" over all multiples of 5 (or 6) and nearby numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/doubling_chain_concepts.py`

### membership_criterion_threshold — pruned (experiment1 end of experiment)

**Margin:** 34.2 nats behind gamma_width_parity_range_concepts (4.6× dse)

**Hypothesis:** People infer how likely the target is to belong to the hidden concept exactly as the current best model's Bayesian concept learner does (same rules and magnitude ranges, family-organised rule prior, size principle), but they do not report that probability directly: they answer "yes" only when the inferred membership probability exceeds an internal decision criterion, and this criterion varies noisily from judgment to judgment. So targets with middling membership probability are mostly rejected (or mostly accepted, depending on where the criterion sits), and answers are more all-or-none than probability matching predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/membership_criterion_threshold.py`

### win_stay_lose_shift_concept_testing — pruned (experiment1 end of experiment)

**Margin:** 49.0 nats behind gamma_width_parity_range_concepts (5.0× dse)

**Hypothesis:** People learn the concept by sequential hypothesis testing ("win-stay, lose-shift") rather than by weighing all examples at once: on reading the first example they pick one concept (a rule or a magnitude range, with the same family-organised rule prior and preferred-width range prior as the best model) in proportion to how well it explains that example, keep it as long as each new example fits it, and only when an example contradicts it do they pick a fresh concept in proportion to how well it explains all examples seen so far. They answer "yes" if the concept they end up holding contains the target, so the first example shapes generalisation most and later examples that fit a broad initial guess do not narrow it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concept_testing.py`

### erlang_interval_size_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 92.2 nats behind gamma_width_parity_range_concepts (7.4× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (prior weight per rule, share lambda_0 for rules, size-principle sampling with 10% random examples, "yes" = posterior probability the concept contains the target). The one change is the prior over intervals: instead of every interval being equally likely (which makes wide ranges a priori far more numerous), people expect a range of a characteristic width, with an Erlang prior over interval size whose scale is inferred from the data (as in Tenenbaum's number game), so moderately narrow ranges are preferred a priori and very wide or single-number ranges are penalised.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/erlang_interval_size_concept_learning.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 95.2 nats behind gamma_width_parity_range_concepts (7.8× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### off_by_one_power_concepts — pruned (experiment2 end of experiment)

**Margin:** 110.6 nats behind mixed_strong_weak_sampling_concepts (5.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (rules with a family-organised prior and arbitrary-exception rules as their own kind, magnitude ranges with a beta prior over the share of the 1–100 line they cover, and the size principle), but their space of rules also contains near-misses of the sparse multiplicative rules: "one more than" or "one less than" a power or a square (e.g. 3, 5, 9, 17, 33, 65 or 1, 3, 7, 15, 31, 63, or 2, 5, 10, 17, 26, 37), forming their own kind of rule with its own inferred, presumably low, plausibility. So after examples such as 31, 3, 1, 15 or 33, 17, 5, 9 people say "yes" to the next shifted power (63, 65) and "no" to other odd numbers or nearby numbers, rather than spreading "yes" over all odd numbers or a range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/off_by_one_power_concepts.py`

### gaussian_neighbourhood_magnitude_concepts — pruned (experiment2 end of experiment)

**Margin:** 211.5 nats behind mixed_strong_weak_sampling_concepts (7.2× dse)

**Hypothesis:** People generalise by rules exactly as the best Bayesian concept learner does, but their magnitude-based concepts are not hard-edged ranges: such a concept is a neighbourhood around a central number, membership falls off smoothly with numerical distance from that centre along a Gaussian similarity gradient whose breadth they infer from the examples (narrow neighbourhoods favoured by the size principle), and examples are produced in proportion to that similarity. A target's graded similarity to the inferred neighbourhood is the probability they say "yes", so "yes" fades gradually with distance from the examples' centre instead of being flat inside the examples' range and stopping at an edge.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/gaussian_neighbourhood_magnitude_concepts.py`

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 213.0 nats behind gamma_width_parity_range_concepts (10.2× dse)

**Hypothesis:** People entertain the same kinds of number concepts (arithmetic rules and magnitude ranges) but treat the examples only as consistent with the hidden concept rather than as random samples drawn from it (weak sampling, no size principle), so every concept that contains the examples stays plausible in proportion to its prior regardless of how many other numbers it includes. The sharpest disagreement with the size-principle best model lives in sets that fit a small rule nested in a larger one (e.g. 16,32,2,8 or 20,32,92): this model predicts broad "yes" to targets that only the larger concept contains (other even numbers, numbers anywhere in a wide range), whereas the best model confines "yes" to the smallest fitting concept.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_learning.py`

### arbitrary_opening_example — pruned (experiment2 end of experiment)

**Margin:** 227.0 nats behind mixed_strong_weak_sampling_concepts (10.1× dse)

**Hypothesis:** People take the first example shown as an arbitrary starting point — just some number the hidden concept contains, picked without regard to how many numbers the concept holds — and treat only the examples that follow as drawn at random from the concept, so only those later examples make them favour small concepts (the size principle). Otherwise they do the same Bayesian concept learning as the best model (rules with a family-organised prior and exception rules as their own kind, ranges judged as a typical share of the 1–100 line, mixed strong/weak sampling for the later examples). So a single example leads to broad generalisation, each further example narrows it sharply, and the same numbers in a different order give different answers depending on which one came first.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/arbitrary_opening_example.py`

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Margin:** 264.6 nats behind gamma_width_parity_range_concepts (9.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_mixture.py`

### generative_next_output_likelihood — pruned (experiment2 end of experiment)

**Margin:** 279.1 nats behind mixed_strong_weak_sampling_concepts (10.5× dse)

**Hypothesis:** People read "would the program produce this number?" as a question about how likely the program is to output that particular number next, not about whether the number belongs to the hidden concept. They infer the concept as a Bayesian concept learner does (family-organised rules with exception rules as their own kind, magnitude ranges covering a typical share of the 1–100 line, the size principle), but they say "yes" according to how much more likely the target is as the next output than a number picked at random. So targets that only a broad concept contains (any even number, anywhere in a wide range) get fewer "yes" answers than their membership probability implies, while members of a small, well-fitting concept (powers of 2, a tight range) are accepted readily.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/generative_next_output_likelihood.py`

### exemplar_featural_similarity_gcm — pruned (experiment1 end of experiment)

**Margin:** 400.3 nats behind gamma_width_parity_range_concepts (14.7× dse)

**Hypothesis:** People do not infer a rule at all: they judge the target by its exemplar similarity to the examples shown, as in a generalized context model. Each example is compared with the target on a few perceptual dimensions of the number — its magnitude, its final digit, its tens digit and its parity — with attention shared among these dimensions, similarity decays exponentially with attention-weighted mismatch, and the more similar the target is on average to the examples, the more likely they say yes.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_featural_similarity_gcm.py`

### property_matching_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 404.0 nats behind gamma_width_parity_range_concepts (14.8× dse)

**Hypothesis:** People do not infer a hidden concept: they store the shown examples as exemplars and judge the target by how similar it is to them, where two numbers are similar to the extent that they share familiar mathematical properties (parity, primality, being a square or cube, being a power of some base, being a multiple of some number, final digit, digits contained) and are close in magnitude. Each kind of property receives its own attention weight, similarity decays exponentially with the attention-weighted count of properties on which target and example differ, and people say "yes" with a Luce-choice probability that grows with the target's average similarity to the stored examples relative to a fixed response criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/property_matching_exemplar_similarity.py`

### residue_class_progression — pruned (experiment1 end of experiment)

**Margin:** 457.7 nats behind gamma_width_parity_range_concepts (12.1× dse)

**Hypothesis:** People assume the hidden program produces an arithmetic progression: all numbers in 1-100 that leave the same remainder when divided by some step size (step 1 = every number, step 2 = all odds or all evens, step 10 = a shared final digit, step 12 = every twelfth number, and so on). They consider every step size whose residue class contains all the examples, favour smaller steps a priori but larger (tighter) steps when several examples all fit them (the size principle), and say "yes" with the posterior probability that the target lies in the same progression as the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/residue_class_progression.py`

### spread_scaled_distance_gradient — pruned (experiment1 end of experiment)

**Margin:** 459.8 nats behind gamma_width_parity_range_concepts (16.0× dse)

**Hypothesis:** People judge the target purely by its numerical closeness to the nearest example on the 1-100 number line, with a Shepard-style exponential generalization gradient whose width is set by how spread out the examples are: they say "yes" with a probability that falls off exponentially with the target's distance from the nearest example, and widely spaced examples (large average gap between neighbouring examples) license a proportionally broader gradient than tightly clustered ones.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment1/model_loop/models/pruned/spread_scaled_distance_gradient.py`

### tversky_prototype_feature_similarity — pruned (experiment2 end of experiment)

**Margin:** 520.0 nats behind mixed_strong_weak_sampling_concepts (15.5× dse)

**Hypothesis:** People judge the target by its similarity to the prototype of the examples — the set of familiar number properties (parity, being a multiple of some number, being a square, cube or power, being prime, final digit, decade) that all the examples share, plus their place on the number line — using Tversky's ratio-of-features similarity: the salience-weighted properties the target has in common with that prototype, relative to those plus the prototype's properties the target lacks and the target's own properties the prototype lacks, with numerical closeness to the nearest example falling off exponentially and counting as one more weighted property. The more similar the target is to the prototype by this metric, the more likely they say "yes", so a target is rejected both for missing what the examples share and for having salient properties of its own that the examples do not.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/tversky_prototype_feature_similarity.py`

### exemplar_concept_similarity_pooling — pruned (experiment2 end of experiment)

**Margin:** 810.7 nats behind mixed_strong_weak_sampling_concepts (19.7× dse)

**Hypothesis:** People do not infer one concept that explains all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars, where the similarity of two numbers is how readily a familiar number concept (a rule such as "multiples of 9", or a magnitude range of typical width) suggested by the exemplar would also contain the target, with more specific shared concepts counting for more. So a target strongly similar to just one example (e.g. a number close to only one of several widely spread examples, or an odd number after 3, 9, 27, 81) still gets a fair share of "yes" answers, and adding consistent examples narrows generalisation much less than a learner who requires one concept to cover them all.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/pruned/exemplar_concept_similarity_pooling.py`
