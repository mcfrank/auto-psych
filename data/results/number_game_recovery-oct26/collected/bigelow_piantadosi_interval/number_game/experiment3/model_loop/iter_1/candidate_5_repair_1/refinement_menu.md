# Refinement menu

The models you may refine, other than the incumbent `inflated_size_familiar_rule_prior`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### volatile_changepoint_inflated_size — rank 1, 0.6 ± 3.1 nats behind the best (0.2× dse: statistically tied with the best), ELPD-LOO -3425.7

**Hypothesis:** Refinement of volatile_concept_changepoint: people still read the examples one at a time, updating a Bayesian belief over weighted rules and ranges while allowing that the program may switch concept between outputs (with everything else as in that model), but in the size principle they do not take a concept's size at face value — they perceive every concept as larger than it is by an inferred constant number of members. The one change, taken from inflated_concept_size_principle, is this perceived-size inflation, which shrinks the evidential advantage of tiny rules like "powers of 2" over larger consistent ones like "even numbers", so after all-power-of-2 sets people generalize to the remaining powers of 2 less sharply relative to other even numbers than the volatile-concept model predicts; zero inflation recovers it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/volatile_changepoint_inflated_size.py`

### volatile_range_width_inflation — rank 2, 8.5 ± 3.6 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3433.5

**Hypothesis:** Refinement of volatile_changepoint_inflated_size (statistically tied with the best model): people still read the examples one at a time, updating a Bayesian belief over 42 weighted rules (with a rule-size prior) and all ranges of 1-100 while allowing that the program may switch concept between outputs (with a stronger- or weaker-weighted first example, a decision gain and a shifted criterion), and they still perceive arithmetic rules as larger than they really are by an inferred number of members. The one change is how people discount the evidence for narrow magnitude ranges: instead of applying their own weaker (or stronger) evidence weight to ranges, they perceive every range as wider than it is by its own inferred number of extra members, and rules and ranges share one evidence weight. A uniform weight scales the advantage of a narrow range at every width, whereas added width wipes it out among narrow ranges and keeps it for wide ones. So a tight range just around clustered examples (52, 57, 55) loses most of its edge over a moderately wider one, and generalization reaches nearby numbers (48, 62), while very wide ranges are still rejected. With equal inflations for rules and ranges, the model becomes volatile_changepoint_inflated_size with the range evidence weight tied to the rule weight.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/volatile_range_width_inflation.py`

### conservative_partial_belief_updating — rank 3, 17.4 ± 6.1 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3442.5

**Hypothesis:** People read the examples one at a time and revise their belief about the program's concept after each one, but conservatively: after the first example (judged by the same Bayesian concept learning over 42 weighted rules with a size-dependent prior and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger- or weaker-weighted first example, a decision gain and shifted criterion), each later example moves their belief only an inferred fraction of the way from what they believed before toward what full Bayesian updating on that example would give. Because the old belief is kept as a share of the new one, broad concepts that fit the early examples (e.g. "even numbers" or a wide range after 16) retain belief even after later examples single out a tiny rule ("powers of 2" after 16, 8, 64), so people generalize to the remaining powers of 2 less sharply relative to other even numbers, and the earlier examples shape belief more than the later ones; an update fraction of one recovers full Bayesian updating.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/conservative_partial_belief_updating.py`

### inflated_concept_size_principle — rank 4, 18.9 ± 8.2 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3444.0

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted rules and all ranges of 1-100, evidence tempered separately for rules and ranges, a stronger-weighted first example, a decision gain and shifted criterion), but they do not take a concept's size at face value in the size principle: they perceive every concept as larger than it is by an inferred constant number of members, so tiny concepts such as "powers of 8" (8, 64) or "powers of 4" (4, 16, 64) lose most of their advantage over moderately larger ones while the contrast between moderate and broad concepts is largely kept. The sharpest disagreement with the best model lives on sets that fit a tiny rule nested inside a larger one (8, 64; 64, 4; 16, 8; 16, 32, 2, 8): the best model generalizes tightly to the tiny rule's other members (16 after 64, 4), whereas this model predicts broader generalization to the larger rule (multiples of 8, ends in 4, even numbers) and fewer yeses to the tiny rule's own members.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/inflated_concept_size_principle.py`

### volatile_concept_changepoint — rank 5, 20.6 ± 6.7 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3445.7

**Hypothesis:** People read the examples one at a time and update their belief about the program's concept after each one, but they believe the program might switch to a different concept between outputs: before each new example they allow, with an inferred switch probability, that the concept has been re-drawn from their prior. This belief in a volatile generator makes recent examples count more than earlier ones in a graded, Bayes-consistent way — a concept supported only by the last few examples keeps substantial belief even when an earlier example contradicts it — while everything else (weighted rules and ranges, size principle with tempered evidence, first-example weighting, rule-size prior, decision gain and criterion) is as in the best model; a switch probability of zero recovers it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/volatile_concept_changepoint.py`

### log_compressed_range_sampling — rank 6, 21.7 ± 6.0 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3446.8

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted rules and all numeric ranges of 1-100, size principle with evidence tempered separately for rules and ranges, a stronger-weighted first example, a rule-size prior, a decision gain and shifted criterion), but they think about magnitude ranges on a compressed, logarithmic-like mental number line: they believe a range's examples are spread evenly in that compressed space, so the "size" of a range is its compressed width (by an inferred degree of compression) rather than its count of numbers. Because high numbers are crowded together on such a line, a range like 80-100 counts as small and examples there support a tight range, while among small numbers the same linear spread counts as wide, so generalization from small examples reaches proportionally further (e.g. 2, 4 generalizes to 7 or 10) and generalization from large clustered examples stays narrow in absolute terms; no compression recovers the best model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/log_compressed_range_sampling.py`

### rule_size_prior_anchored_evidence — rank 7, 23.3 ± 5.8 nats behind the best (4.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3448.4

**Hypothesis:** Refinement of first_example_evidence_weighting: people do the same Bayesian concept learning over 42 weighted rules and all ranges of 1-100 (evidence tempered separately for rules and ranges, the first example weighed as a stronger or weaker draw than later ones, a decision gain and shifted criterion), but before seeing any examples they do not find tiny, exotic rules ("powers of 4", "powers of 8", "cubes") as plausible as larger, familiar ones ("even numbers", "ends in 4", "multiples of 8"). The one change is that each rule's prior belief rises or falls as an inferred power of how many numbers it contains, which counteracts the size principle for small rules independently of set size, so after 64, 4 people generalize to 14, 24, 44 ("ends in 4", "even") rather than committing to 16 ("powers of 4") and after power-of-2 sets they leak to other even numbers more than the incumbent predicts; an exponent of zero recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/rule_size_prior_anchored_evidence.py`

### volatile_log_compressed_ranges — rank 8, 25.3 ± 7.6 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3450.4

**Hypothesis:** Refinement of log_compressed_range_sampling: people do the same Bayesian concept learning over 42 weighted rules (with a rule-size prior) and all ranges of 1-100 (evidence tempered separately for rules and ranges, a stronger- or weaker-weighted first example, a decision gain and shifted criterion), and they still think of magnitude ranges on a compressed, logarithmic-like mental number line, so a range's examples are believed spread evenly in compressed space and ranges among high numbers count as small while ranges among low numbers count as wide. The one change, taken from volatile_concept_changepoint, is that they read the examples one at a time and believe the program may switch concept between outputs (with an inferred switch probability, re-drawing the concept from their prior), so recent examples count more than earlier ones and a concept supported only by the last examples keeps belief; zero switch probability recovers log_compressed_range_sampling and zero compression approximately recovers volatile_concept_changepoint. The critique shows people generalize to the remaining powers of 2 less sharply than the incumbent predicts; on a compressed line powers of 2 are evenly spaced, so after power-of-2 sets a compressed magnitude range spanning the examples competes with "powers of 2" and pulls yeses toward nearby non-power even (and odd) numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/volatile_log_compressed_ranges.py`

### prototype_property_profile_match — rank 9, 291.6 ± 45.9 nats behind the best (6.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -3716.7

**Hypothesis:** People do not infer which concept generated the examples; instead they abstract a prototype of the example set — a profile of how strongly the set as a whole has each familiar number property (each arithmetic rule such as "powers of 2" or "ends in 4", and each numeric range), given by the fraction of the examples that have it, with rare, distinctive properties more salient than common ones. A target is endorsed in proportion to how much of this salience-weighted prototype it matches, so the salience of a small property is fixed rather than growing with each extra example (no size principle compounding over examples), and properties shared by only some of the examples still contribute partial support.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/prototype_property_profile_match.py`

### place_value_digit_set_induction — no comparison row

**Hypothesis:** People do not think about these numbers arithmetically or as magnitudes on a number line; they treat each number as a two-digit written string and infer which tens digits and which final digits the program is allowed to write, favouring the smallest digit sets that cover the examples (the size principle over a tens-digit-by-final-digit grid). A target is endorsed to the degree that both its tens digit and its final digit are believed allowed, so after 2, 8 people accept 62 or 88 (seen final digits) but reject 16, and after 64, 4 they accept 14, 24, 94 but not 16, generalizing by shared digits rather than by powers or multiples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/place_value_digit_set_induction.py`

### rule_membership_recognition_failure — no comparison row

**Hypothesis:** People infer the program's concept as the best model does (Bayesian weighing of weighted rules and numeric ranges, with inflated perceived concept sizes and a size-dependent rule prior), but when they check whether a new target belongs to a rule they believe in, the check can fail. Unseen members of small, sparse rules are often not recognised as members ("is 32 a power of 2?", "is 16 a power of 4?"), while members of large, dense rules are ("is 48 even?", "does 24 end in 4?"). Recognition rises with how many numbers the rule contains, and examples they were shown are always recognised. This disagrees most sharply with the best model on the unseen members of tiny rules after sets that fit them (16 or 32 after 8, 64; 16 after 64, 4 or 2, 8): even when people believe "powers of 2", they say yes to the remaining powers of 2 less often than to other even numbers. Ranges are checked exactly, and perfect recognition recovers the best model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/rule_membership_recognition_failure.py`

### calculated_rule_check_lapses — no comparison row

**Hypothesis:** Refinement of inflated_size_familiar_rule_prior. People infer the program's concept exactly as in that model: Bayesian concept learning over 42 rules, with a prior that rises as a power of rule size, and all ranges of 1-100, with perceived sizes inflated by a constant, tempered evidence, a re-weighted first example, a decision gain and a shifted criterion. The one change is in how they then check a new target against a rule they believe in. For rules that can be read off the digits (even, odd, ends in or contains a digit, multiples of 5, 10 or 11, powers of 10), the check never fails. For rules that need arithmetic (powers, other multiples, squares, cubes, primes), the check succeeds only with an inferred probability. Targets that were shown as examples are always recognized. So even people who believe "powers of 2" or "powers of 4" give fewer yeses to that rule's unseen members than to other even numbers or same-final-digit numbers (16 after 64, 4 or after 2, 8, compared with 14, 24 or 62). This addresses the critique that the incumbent generalizes too readily to the remaining powers of 2. A check success rate of one recovers the incumbent. Unlike the earlier fallible-verification proposal, which was never scored because it was too slow on a volatile base, failures here depend on whether the check needs calculation, not on the rule's size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment3/model_loop/models/calculated_rule_check_lapses.py`

## Pruned models (out of the set; narrowest margin first)

### win_stay_lose_shift_decision_gain — pruned (experiment1 end of experiment)

**Margin:** 9.6 nats behind criterion_shifted_redundant_examples; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 9

**Hypothesis:** Refinement of win_stay_lose_shift_concept_tracking: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it), but they do not answer by occasionally guessing at random. The one change, taken from the incumbent, replaces the flat guessing floor with an inferred decision gain on the log-odds of their belief that the tracked concept contains the target: people hedge (or sharpen) symmetrically between belief and answer, so targets with near-zero belief under the order-dependent guess still get near-zero yeses.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_decision_gain.py`

### rule_size_inflation_anchored — pruned (experiment2 end of experiment)

**Margin:** 10.6 nats behind volatile_changepoint_inflated_size (2.1× dse)

**Hypothesis:** Refinement of first_example_evidence_weighting: people do the same Bayesian concept learning over weighted rules and numeric ranges (evidence tempered separately for rules and ranges, the first example weighed as an extra-strong or weak draw, a decision gain and shifted criterion), but they do not judge a mathematical rule's size by exactly counting its members — they treat every rule as if it had an inferred number of extra members beyond those it actually has in 1-100. The one change is this subjective size inflation of rules, which shrinks the size-principle advantage of tiny rules ("powers of 2", "powers of 4") over broader ones ("even numbers", "ends in 4", "multiples of 8") — and shrinks it more the more examples are shown — so after 64, 4 or 2, 8 people generalize to other even numbers or same-final-digit numbers more readily than to the remaining powers; zero extra members recovers the refined model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/rule_size_inflation_anchored.py`

### offset_progression_concepts — pruned (experiment1 end of experiment)

**Margin:** 12.3 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted mathematical rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain turning belief into a yes/no answer), but their space of rules also contains offset arithmetic progressions — "every k-th number starting from r" (numbers leaving remainder r when divided by k, for steps 3 to 12), such as 1, 7, 13, 19, ... — and not only multiples counted from zero. These progressions share an inferred, a-priori small slice of prior belief, equally divided among them, so a set whose examples happen to fall on a common step (e.g. 85, 19, 91 all one more than a multiple of 6) generalizes to the other members of that progression.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/offset_progression_concepts.py`

### tempered_evidence_decision_gain — pruned (experiment1 end of experiment)

**Margin:** 13.6 nats behind criterion_shifted_redundant_examples (2.1× dse)

**Hypothesis:** Refinement of decision_gain_concept_learning: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples) and turn their belief into a yes/no answer through the same inferred decision gain on the log-odds, but they do not weigh the examples as fully independent draws from the concept — the one change, taken from the incumbent, is that each example's evidence counts for an inferred fraction (or multiple) of a full draw. Tempering thus decides which concepts people entertain (how broadly they generalize), while the decision gain alone decides how they hedge between belief and answer, with no flat guessing floor: targets with near-zero belief still get near-zero yeses, unlike under the incumbent's guessing.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_decision_gain.py`

### win_stay_lose_shift_concept_tracking — pruned (experiment1 end of experiment)

**Margin:** 23.3 nats behind criterion_shifted_redundant_examples (2.7× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they hold a single running guess that they revise example by example, in the order shown (win-stay, lose-shift). After the first example they pick a concept in proportion to its Bayesian plausibility (weighted rules and intervals, size principle), keep it as long as each new example fits it, and only when an example contradicts it do they switch to a fresh concept drawn from the posterior given all examples seen so far — so broad concepts adopted early survive later examples that a full Bayesian learner would use to narrow them, and the same numbers in a different order generalize differently.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_shift_concept_tracking.py`

### redundant_examples_tempered_concepts — pruned (experiment1 end of experiment)

**Margin:** 25.0 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of tempered_evidence_concept_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with tempered evidence and occasional guessing, but they treat additional examples as partly redundant rather than as independent draws, so the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it. The one change is this diminishing (or accelerating) return to set size: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen it much less than strong sampling predicts; an exponent of one recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/redundant_examples_tempered_concepts.py`

### tempered_evidence_concept_guessing — pruned (experiment1 end of experiment)

**Margin:** 27.4 nats behind criterion_shifted_redundant_examples (2.9× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100, with the same occasional guessing, but they do not weigh the examples as fully independent random draws from the concept — each example's evidence counts for an inferred fraction (or multiple) of a full draw. The one change is this tempered size principle: weighting the evidence below one keeps broad rules and wide intervals plausible and widens generalization, while weighting it above one makes people lock onto the tightest concept consistent with the examples.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_evidence_concept_guessing.py`

### first_example_evidence_weighting — pruned (experiment2 end of experiment)

**Margin:** 28.4 nats behind volatile_changepoint_inflated_size (3.3× dse)

**Hypothesis:** Refinement of first_example_anchored_concepts: people still let the first example they see anchor their Bayesian concept learning over weighted rules and numeric ranges (with evidence tempered separately for rules and ranges, and a decision gain and shifted criterion), but the anchoring works through evidence rather than a flat bonus: the first example is weighed as a stronger (or weaker) draw from the concept than later examples, by an inferred multiplier on its size-principle evidence. The one change replaces the incumbent's equal boost to every concept containing the first example with this extra evidential weight, so the first example pulls belief most strongly toward small, specific concepts that contain it (e.g. "powers of 2" when 16 is shown first) and only weakly toward broad ones like "even numbers" or wide ranges, while concepts that must write it off as noise are penalised more.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_example_evidence_weighting.py`

### anchored_evidence_bounded_rules — pruned (experiment2 end of experiment)

**Margin:** 28.8 nats behind volatile_changepoint_inflated_size (3.4× dse)

**Hypothesis:** Refinement of anchored_range_bounded_rules: people do the same Bayesian concept learning over 42 weighted rules, their range-bounded runs (e.g. "multiples of 10 from 70 to 100", or "powers of 2 up to 16") and all ranges of 1-100, with evidence tempered separately for arithmetic concepts and for plain ranges, and answer through the same decision gain and shifted criterion; the first example shown still anchors the inference. The one change is how it anchors: instead of a flat belief boost for every concept containing the first example, the first example is weighed as a stronger (or weaker) draw from the concept than later examples, by an inferred multiplier on its size-principle evidence, so the anchor pulls belief most toward the tightest concepts that contain it — a short bounded run around the first example rather than the whole rule — which makes people generalize less to rule members far from the examples (e.g. 64 after 2, 4, 8) than the unbounded-rule incumbent predicts.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/anchored_evidence_bounded_rules.py`

### first_example_anchored_concepts — pruned (experiment2 end of experiment)

**Margin:** 31.0 nats behind volatile_changepoint_inflated_size (3.3× dse)

**Hypothesis:** People infer the program's concept by the same Bayesian weighing of mathematical rules and numeric ranges as the best model, but the first example they see anchors that inference: they treat it as the defining instance of the program, so every concept containing the first example gets extra belief, while concepts that would have to dismiss it as a stray number are disfavoured more than concepts that dismiss a later example. The same numbers shown in a different order therefore generalize differently — an odd-one-out shown first pulls belief toward concepts that include it, while the same odd-one-out shown last is readily written off.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/first_example_anchored_concepts.py`

### anchored_range_bounded_rules — pruned (experiment2 end of experiment)

**Margin:** 31.8 nats behind volatile_changepoint_inflated_size (3.5× dse)

**Hypothesis:** Refinement of range_bounded_rules_split_tempering: people do the same Bayesian concept learning over 42 weighted rules, their range-bounded runs (e.g. "multiples of 10 from 70 to 100") and all ranges of 1-100, with evidence tempered separately for arithmetic concepts and for plain ranges, and answer through the same decision gain and shifted criterion. The one change, taken from first_example_anchored_concepts, is that the first example shown anchors the inference: every concept that contains it (any rule, bounded run or range) gets an inferred boost in belief, while concepts that would have to write it off as a stray number are disfavoured, so the same numbers in a different order generalize differently and a tight, range-bounded rule built around the first example is favoured; zero boost recovers the refined model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/anchored_range_bounded_rules.py`

### split_rule_range_tempering — pruned (experiment2 end of experiment)

**Margin:** 32.8 nats behind volatile_changepoint_inflated_size (3.5× dse)

**Hypothesis:** Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and answer through the same decision gain and shifted criterion, but they do not weigh the examples' evidence equally strongly for the two kinds of concept. The one change is that the size principle is tempered separately, by its own inferred weight, for mathematical rules and for numeric ranges: people may treat examples as strong, informative samples of an arithmetic rule (locking onto "powers of 2" after 16, 8, 64) while treating them as only weak evidence about where a magnitude range starts and stops (so range-based generalization stays broad), or vice versa; equal weights recover the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/split_rule_range_tempering.py`

### range_bounded_rules_split_tempering — pruned (experiment2 end of experiment)

**Margin:** 33.3 nats behind volatile_changepoint_inflated_size (3.6× dse)

**Hypothesis:** Refinement of range_bounded_rule_concepts: people do the same Bayesian concept learning over 42 weighted rules, their range-bounded versions (any unbroken run of a rule's members, e.g. "multiples of 10 from 70 to 100", taking an inferred share of each rule's prior) and all ranges of 1-100, and answer through the same decision gain and shifted criterion. The one change, taken from split_rule_range_tempering, is that the examples' evidence is tempered by separate inferred weights for arithmetic concepts (rules and their bounded runs) and for plain magnitude ranges, so people can lock onto a tight, possibly range-bounded rule from a few examples while still treating the same examples as weak evidence about where a plain range stops (equal weights recover range_bounded_rule_concepts).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_bounded_rules_split_tempering.py`

### width_prior_split_tempering — pruned (experiment2 end of experiment)

**Margin:** 33.8 nats behind volatile_changepoint_inflated_size (3.7× dse)

**Hypothesis:** Refinement of split_rule_range_tempering: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, evidence tempered separately for rules and for ranges) and answer through the same decision gain and shifted criterion, but they do not consider all numeric ranges equally plausible before seeing any examples. The one change is that a range's prior belief rises or falls as an inferred power of its width, so people may favour broad magnitude ranges (or narrow ones) a priori, independently of how many examples they have seen — unlike evidence tempering, whose effect grows with set size; an exponent of zero recovers the refined model.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/width_prior_split_tempering.py`

### exemplar_rule_member_similarity — pruned (experiment2 end of experiment)

**Margin:** 35.7 nats behind volatile_changepoint_inflated_size (3.5× dse)

**Hypothesis:** People store each familiar arithmetic category ("powers of 2", "multiples of 10", "ends in 3", ...) as a list of remembered member numbers, and judge both the examples and the target against a category by exemplar similarity to its nearest stored member on the number line, not by all-or-none membership; the similarity falls off geometrically with distance at an inferred rate. Categories whose stored members closely match all the examples (diluted by how many members the category stores) are retrieved most strongly, alongside plain numeric ranges, and the target is endorsed by its similarity to the members of the retrieved categories — so near-misses of a rule (30 or 66 after 16, 8, 64) get graded partial yeses rather than none.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/exemplar_rule_member_similarity.py`

### parity_restricted_rules — pruned (experiment2 end of experiment)

**Margin:** 36.4 nats behind volatile_changepoint_inflated_size (3.7× dse)

**Hypothesis:** People do the same Bayesian concept learning over weighted mathematical rules and numeric ranges as the best models (size principle with tempered evidence, a decision gain and shifted criterion turning belief into a yes/no answer), but their space of rules also contains each rule restricted to odd or to even numbers — "odd multiples of 3", "even squares", "odd numbers containing a 7" — which take an inferred share of their parent rule's prior belief. Because such a parity-restricted rule is half the size, the size principle favours it whenever all the examples share a parity, so after 63, 27, 99, 33 people endorse other odd multiples of 3 but reject even ones like 18 or 24.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/parity_restricted_rules.py`

### criterion_shifted_tempered_decision — pruned (experiment2 end of experiment)

**Margin:** 37.0 nats behind volatile_changepoint_inflated_size (3.7× dse)

**Hypothesis:** Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, examples' evidence tempered by an inferred weight) and pass their belief through the same inferred decision gain on its log-odds, but their yes/no criterion is not centred on even odds. The one change, taken from response_bias_concept_learning, is an inferred response bias that shifts the decision criterion by a constant on the log-odds scale, so the gain decides how sharply people separate strong from weak beliefs while the bias decides whether they lean toward "yes" or "no" — e.g. a liberal criterion gives more yeses to targets with middling or weak belief without making yeses to the examples themselves less likely.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/criterion_shifted_tempered_decision.py`

### round_landmark_interval_prior — pruned (experiment2 end of experiment)

**Margin:** 38.4 nats behind volatile_changepoint_inflated_size (3.9× dse)

**Hypothesis:** People do the same Bayesian concept learning over weighted mathematical rules and numeric ranges as the best model (size principle with tempered evidence, a decision gain and criterion turning belief into a yes/no answer), but the ranges they entertain are not all equally natural: ranges whose boundaries fall on round decade landmarks (such as 1-50, 20-29, 61-100 or "the 80s") get extra prior belief, by an inferred factor, over ranges with arbitrary edges like 23-87. Generalization from examples therefore tends to stop at, or extend out to, the nearest round decade boundary rather than tracking the examples' exact span.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/round_landmark_interval_prior.py`

### stochastic_example_forgetting — pruned (experiment2 end of experiment)

**Margin:** 39.2 nats behind volatile_changepoint_inflated_size (4.0× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (weighted rules and all intervals of 1-100, size principle with tempered evidence, and a decision gain and criterion turning belief into a yes/no answer), but their memory of the examples is unreliable: by the time they judge a target, each example has been retained only with an inferred probability, and they infer the concept from whichever examples they still hold. Their belief is therefore an average over the possible remembered subsets of the set, so larger sets are sometimes judged as if from fewer examples (broader generalization) and an odd-one-out example is sometimes forgotten, letting the tight rule the other examples share (e.g. "ends in 2" after 92, 82, 27, 42) survive without being explained away as noise.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/stochastic_example_forgetting.py`

### range_bounded_rule_concepts — pruned (experiment2 end of experiment)

**Margin:** 39.7 nats behind volatile_changepoint_inflated_size (4.0× dse)

**Hypothesis:** People do the same Bayesian concept learning as the best model (42 weighted rules and all intervals of 1-100, size principle with tempered evidence, a decision gain and response bias turning belief into a yes/no answer), but their rules can also be bounded to a range: besides "multiples of 10" they entertain "multiples of 10 from 70 to 100", i.e. any unbroken run of a rule's members, which takes an inferred share of each rule's prior belief. Because a range-bounded rule is smaller, the size principle favours it whenever the examples cluster, so this model disagrees most sharply with the best model on rule-consistent targets far outside the examples' range (e.g. 10 after 80, 70, 90, 100, or 6 after 96, 86, 36, 66), which it predicts far fewer yeses to, while keeping rule-violating targets inside the range at near-zero.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/range_bounded_rule_concepts.py`

### decision_gain_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 49.7 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples), but they do not report their belief that the target is in the concept directly — they turn it into a yes/no decision with an inferred decision gain on the odds. The one change is this gain: above one, people maximize (beliefs above even odds become near-certain yeses, below even odds near-certain noes); below one, they hedge, so weak beliefs (targets just outside plausible concepts) get more yeses and strong beliefs (the examples themselves) fewer, symmetrically in log-odds rather than as a flat guessing floor.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decision_gain_concept_learning.py`

### inferred_example_noise_belief — pruned (experiment1 end of experiment)

**Margin:** 50.9 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** People do Bayesian concept learning over weighted mathematical rules and number intervals with the size principle, but they distort one assumption of the normative learner: how often they believe an example is a random number not produced by the concept. That believed noise rate is a free, inferred quantity rather than a fixed small value; the more examples people discount as noise, the more readily they accept rules that fit only some of the examples (e.g. "ends in 2" after 92,82,27,42) and the broader the intervals they entertain.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/inferred_example_noise_belief.py`

### example_noise_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 51.7 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of concept_learning_with_guessing: people do the same Bayesian concept learning over 42 weighted rules and all intervals of 1-100 (size principle) with occasional guessing, but how readily they treat an example as a stray "noise" number that does not belong to the concept is a property of the learner, inferred from the data, rather than fixed at 10%. The one change is this inferred example-noise rate: a high rate lets a rule survive an example that breaks it (e.g. "ends in 2" after 92, 82, 27, 42) and a low rate forces a concept that covers every example, so it changes which concept people infer from sets containing an outlier.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/example_noise_concept_learning.py`

### concept_learning_with_guessing — pruned (experiment1 end of experiment)

**Margin:** 53.1 nats behind criterion_shifted_redundant_examples (3.4× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over 42 weighted rules and all intervals (size principle, 10% noisy examples), but on a fraction of trials they do not consult the inferred concept at all and instead guess, answering "yes" with their own guessing bias. The one change is this inferred guessing (lapse) rate with a yes-bias, which puts a floor under "yes" for targets far outside every plausible concept (the far-outside-range yeses the incumbent under-produces) and a ceiling below 1 for the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/concept_learning_with_guessing.py`

### serial_position_weighted_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 53.6 nats behind criterion_shifted_redundant_examples (3.2× dse)

**Hypothesis:** People read the examples one at a time and integrate them into the same Bayesian concept learning over rules and intervals (with occasional guessing) as the best model, but their memory and attention are unequal across the sequence: each example's evidence is weighted by its serial position, with a single inferred gradient that favours the most recent examples (leaky memory / forgetting) or the earliest ones (primacy, fading attention). Because earlier and later examples then count differently, generalization from multi-example sets depends on the order in which the same numbers were shown, unlike the order-blind incumbent; a flat gradient recovers it.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/serial_position_weighted_concept_learning.py`

### response_bias_concept_learning — pruned (experiment1 end of experiment)

**Margin:** 55.0 nats behind criterion_shifted_redundant_examples (3.6× dse)

**Hypothesis:** People do the same Bayesian concept learning as the simplest rule-and-interval model (42 weighted rules and all intervals of 1-100, size principle, 10% noisy examples), but they do not report their belief directly: they compare the evidence for "the program makes this number" against a shifted decision criterion, a response bias that adds a constant amount to the log-odds of every belief. A liberal criterion raises yeses most for middling and weak beliefs (targets just outside plausible concepts) while leaving near-certain beliefs nearly unchanged; unlike a guessing floor it does not lower yeses to the examples themselves, and unlike a decision gain it shifts all beliefs in one direction rather than pushing them apart or together.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/response_bias_concept_learning.py`

### criterion_shifted_redundant_examples — pruned (experiment2 end of experiment)

**Margin:** 55.5 nats behind volatile_changepoint_inflated_size (4.7× dse)

**Hypothesis:** Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same decision gain and shifted (biased) criterion on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_decision_gain, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it, so with sublinear growth one example already supports a fairly tight concept while three or four sharpen generalization less than strong sampling predicts, while the criterion shift separately sets the overall lean toward "yes" (an exponent of one recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/criterion_shifted_redundant_examples.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 58.1 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### interval_width_power_prior — pruned (experiment1 end of experiment)

**Margin:** 60.5 nats behind criterion_shifted_redundant_examples (3.5× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people still do Bayesian concept learning over the same 42 weighted rules and all intervals of 1-100 with the size principle, but their prior over intervals is not flat — it rises or falls as a power of the interval's width, with the exponent inferred. A prior that favours wide intervals offsets the size principle and makes people generalize further beyond the range of the examples (the incumbent under-produces "yes" to targets far outside that range); an exponent of zero recovers the incumbent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/interval_width_power_prior.py`

### probability_threshold_decision — pruned (experiment2 end of experiment)

**Margin:** 63.3 nats behind volatile_changepoint_inflated_size (4.9× dse)

**Hypothesis:** People infer the program's concept exactly as the simplest current model does (Bayesian weighing of weighted mathematical rules and numeric ranges, with tempered evidence), but they answer by comparing their degree of belief that the target belongs against a fixed certainty threshold on the plain probability scale, with comparison noise of constant size in probability units. Because the noise is constant in probability rather than in log-odds, all weakly believed targets (belief 1% or 15%) get nearly the same small floor of yeses and all strongly believed targets (85% or 99%) nearly the same ceiling, while answers change steeply only for beliefs near the threshold — unlike a gain and criterion on log-odds, which keep distinguishing 1% from 15%.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/probability_threshold_decision.py`

### redundant_examples_decision_gain — pruned (experiment2 end of experiment)

**Margin:** 67.0 nats behind volatile_changepoint_inflated_size (4.5× dse)

**Hypothesis:** Refinement of tempered_evidence_decision_gain: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and turn their belief into a yes/no answer through the same inferred decision gain on its log-odds, but they treat additional examples as partly redundant rather than as independent draws. The one change, taken from redundant_examples_tempered_concepts, is that the total evidence from a set grows as an inferred power of the number of examples instead of in proportion to it: with sublinear growth, a single example already supports a fairly tight concept while three or four examples sharpen generalization much less than strong sampling predicts (an exponent of one recovers the incumbent).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/redundant_examples_decision_gain.py`

### strong_weak_sampling_mixture — pruned (experiment2 end of experiment)

**Margin:** 83.6 nats behind volatile_changepoint_inflated_size (4.8× dse)

**Hypothesis:** Refinement of split_rule_range_tempering: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (10% noisy examples), judge the evidence separately for rules and for ranges, and answer through the same decision gain and shifted criterion. The one change is how the size principle is weakened: instead of discounting every example's evidence by a power, people believe each example was either drawn from the concept itself or picked at random from 1-100 and merely found to fit it (a mixture of strong and weak sampling, with an inferred strong-sampling share for rules and another for ranges). The size penalty therefore saturates — tiny concepts are still sharply distinguished from each other, but a large concept consistent with the examples (e.g. "multiples of 8" after 8, 64) is barely penalised relative to a tiny one, so generalization to larger consistent rules and wide ranges stays strong.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/strong_weak_sampling_mixture.py`

### weak_sampling_concept_elimination — pruned (experiment1 end of experiment)

**Margin:** 95.3 nats behind criterion_shifted_redundant_examples (5.5× dse)

**Hypothesis:** People generalize by weak-sampling hypothesis elimination: they entertain the same kinds of candidate concepts (mathematical rules and numeric intervals) but treat the examples only as positive labels on numbers that happened to come up, not as draws from the concept, so every concept consistent with the examples stays equally plausible regardless of its size (no size principle). "Yes" is the prior-weighted share of the still-consistent concepts that contain the target, with an occasional example tolerated as an error; small, tight concepts get no advantage, so generalization is far broader than under strong sampling.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/weak_sampling_concept_elimination.py`

### predictive_representativeness_judgment — pruned (experiment1 end of experiment)

**Margin:** 96.9 nats behind criterion_shifted_redundant_examples (4.5× dse)

**Hypothesis:** People do not ask whether the target belongs to the concept they have inferred; they ask how representative the target would be as the program's next output — how much more likely the program is to produce exactly this number than a number picked at random, given the concepts the examples suggest (weighted rules and intervals, size principle). Their "yes" rises smoothly with this predicted probability of the target being generated next, so a target in a small, tight concept (32 after 16, 8, 64) is endorsed more readily than an equally certain member of a large concept (a number after examples suggesting "even" or a wide interval), unlike a membership judgment that treats all members of a believed concept alike.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/predictive_representativeness_judgment.py`

### win_stay_lose_shift_criterion_shift — pruned (experiment2 end of experiment)

**Margin:** 110.2 nats behind volatile_changepoint_inflated_size (7.5× dse)

**Hypothesis:** Refinement of win_stay_lose_shift_decision_gain: people still hold a single running guess about the concept (drawn from the tempered Bayesian posterior over weighted rules and intervals after the first example, kept while later examples fit it, redrawn from the current posterior when an example contradicts it) and pass their belief that this tracked concept contains the target through an inferred decision gain on its log-odds. The one change, taken from the incumbent criterion_shifted_tempered_decision, is an inferred response bias that shifts the yes/no criterion away from even odds by a constant on the log-odds scale, so a liberal (or conservative) criterion raises (or lowers) yeses to targets with middling or weak belief under the order-dependent guess without much changing yeses to the examples themselves.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/win_stay_lose_shift_criterion_shift.py`

### tempered_size_principle_rules_intervals — pruned (experiment1 end of experiment)

**Margin:** 216.6 nats behind criterion_shifted_redundant_examples (6.8× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over mathematical rules and intervals as in that model, but they discount the evidence of the examples — each example counts for only a fraction (or a multiple) of a fully independent random draw from the concept, by a fitted evidence weight — so the size principle is tempered and generalization stays broader than strong sampling predicts, reaching targets well outside the examples' range. The single change is this fitted tempering of the likelihood (the strength of the size principle), which the original fixes at full strength.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tempered_size_principle_rules_intervals.py`

### exemplar_averaged_generalization — pruned (experiment1 end of experiment)

**Margin:** 285.1 nats behind criterion_shifted_redundant_examples (8.7× dse)

**Hypothesis:** People do not infer one concept from the whole set of examples; instead each example is stored as a separate exemplar, and the target is judged by how strongly each single stored example on its own generalizes to it — through the familiar number categories (mathematical rules and numeric ranges) that the example and target both belong to, with smaller categories counting more (the size principle applied to one example at a time). These single-example generalization strengths are averaged across the examples, GCM-style, and the average belief is turned into a yes/no answer, so adding consistent examples never sharpens generalization the way joint Bayesian inference does: a target that only some examples point to (e.g. a multiple of 4 after 16, 8, 2, 64) still gets substantial support.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_averaged_generalization.py`

### shared_property_match_fraction — pruned (experiment1 end of experiment)

**Margin:** 353.2 nats behind criterion_shifted_redundant_examples (9.4× dse)

**Hypothesis:** People do not weigh competing concepts or the size of each concept; instead they note every property that all the examples have in common (parity, final digit, tens decade, being a multiple, power, square, cube or prime, lying within the examples' range) and endorse a target in proportion to how many of those shared properties it also has. The more of the examples' common properties the target matches, the more likely a "yes", regardless of how many numbers in 1-100 share those properties.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_match_fraction.py`

### featural_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 354.9 nats behind criterion_shifted_redundant_examples (12.3× dse)

**Hypothesis:** People do not entertain candidate rules at all; instead they judge a target by its featural similarity to each example they saw, treating numbers as bundles of salient properties (parity, divisibility by 3, 4, 5 and 10, last digit, decade, being prime, square or a power of two). Each example's similarity falls off exponentially with the number of properties on which it and the target disagree, and the chance of saying "yes" rises with the average similarity across the examples (an exemplar, GCM-style judgment with no size principle).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/featural_exemplar_similarity.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 362.9 nats behind criterion_shifted_redundant_examples (5.8× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`

### digit_grid_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 445.2 nats behind criterion_shifted_redundant_examples (13.3× dse)

**Hypothesis:** People judge a target by its similarity to the examples in the written-digit form of the numbers: each number is a point on a grid of its tens digit and its ones digit, and a target's similarity to an example falls off exponentially with the attention-weighted city-block distance between them on that grid (with an inferred share of attention given to the ones versus the tens digit). The more similar the target is on average to the examples, the more likely a "yes" — so a target sharing the examples' final digit (e.g. 52 after 92, 82, 42) or decade looks similar even when it is far away numerically, without any rule being entertained.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_grid_exemplar_similarity.py`

### log_magnitude_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 612.2 nats behind criterion_shifted_redundant_examples (15.2× dse)

**Hypothesis:** People ignore arithmetic rules entirely and generalize by exemplar similarity on a compressed (logarithmic) mental number line: the chance of saying "yes" falls off smoothly with the target's distance to the nearest example, and the falloff is wider when the examples themselves are spread far apart. This disagrees most sharply with the rule-based best model on rule-consistent targets far from every example (e.g. 64 after 16,32,2,8 — rules say yes, similarity says unlikely) and on rule-violating targets nestled among the examples (e.g. 20 after 16,32,2,8 — rules say no, similarity says likely).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment1/model_loop/models/pruned/log_magnitude_exemplar_similarity.py`

### conjunctive_category_similarity — pruned (experiment2 end of experiment)

**Margin:** 757.0 nats behind volatile_changepoint_inflated_size (17.2× dse)

**Hypothesis:** People judge a target by its similarity to each example separately, where two numbers are similar to the degree that they fall together in the same familiar number categories (mathematical rules or numeric ranges), with small, specific shared categories counting more than broad ones by an inferred similarity gradient. These per-example similarities combine conjunctively (multiplicatively): the target must resemble every example, so a single example it is dissimilar to sharply lowers the chance of a "yes", and the combined similarity is turned into a yes/no answer through a decision gain and criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/pruned/conjunctive_category_similarity.py`
