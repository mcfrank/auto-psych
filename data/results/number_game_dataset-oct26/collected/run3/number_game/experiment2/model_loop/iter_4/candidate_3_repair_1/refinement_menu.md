# Refinement menu

The models you may refine, other than the incumbent `outlier_tolerant_rule_concepts`. Each entry is one mechanism: its hypothesis in full as its author stated it, how it stands, and where its source is. This is a menu, not a list of what is ruled out: a pruned model lost to the best model on the data it was scored on, by the stated margin, but its mechanism may be partly right — and a model that lost narrowly is the most promising target here. Read the source of the model you pick before you write anything.

## Live models other than the incumbent (in the set; best first)

### evidence_scarce_guessing_concepts — rank 1, 9.5 ± 3.3 nats behind the best (2.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2806.0

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100, with evidence weight growing with set size, but when the examples give them little to go on they often do not consult a concept at all and simply guess (with their own inferred leaning toward "yes"), and this guessing becomes rarer as more examples are shown (an inferred guessing rate for a single example and an inferred decline with set size). The one change is this evidence-dependent guessing, which predicts that after a single example people say "yes" fairly often even to targets no plausible concept contains (e.g. 94 or 100 after 7, 1 after 100), whereas after three or four examples such targets are almost always rejected — the excess single-example non-member "yes" rate the critique of the best model flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/evidence_scarce_guessing_concepts.py`

### family_weighted_evidence_concepts — rank 2, 11.7 ± 5.0 nats behind the best (2.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2808.3

**Hypothesis:** Refinement of `evidence_scarce_guessing_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100 (evidence weight growing with set size, guessing that becomes rarer as more examples are shown), but they do not weigh the examples' evidence equally for the two kinds of concept: how strongly a tight fit counts in a concept's favour (the size principle) differs between arithmetic rules and number ranges, by an inferred ratio. The one change is this family-specific evidence weight, which, if people discount the coincidence that examples cluster in a narrow range relative to the coincidence that they all fit a rule, predicts that sets whose examples share a property (e.g. all even, or all ending in 4) are explained by that rule rather than by a nearby range, so nearby targets lacking the property (e.g. 25 after 22, 26, 30) are rejected more sharply — the excess opposite-parity "yes" rate the critique of the incumbent flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/family_weighted_evidence_concepts.py`

### fuzzy_edged_range_concepts — rank 3, 18.7 ± 8.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2815.3

**Hypothesis:** Refinement of `session_growing_acquiescence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100 (evidence weight growing with set size, blind "yes" growing over the session), but they treat a number range as a vague category with fuzzy edges rather than a crisp one: when checking whether the target falls in a range they believe in, a target just outside its edges still counts as partly inside, with credit fading with distance beyond the edge at an inferred tolerance. The one change is this graded range membership at the decision stage (arithmetic rules stay crisp, and the examples are still explained by crisp ranges), which predicts broader, smoothly decaying generalization to near-but-outside targets — strongest after a single example, whose posterior concentrates on narrow ranges around it — the excess single-example generalization the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/fuzzy_edged_range_concepts.py`

### session_growing_acquiescence_concepts — rank 4, 19.6 ± 8.8 nats behind the best (2.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2816.2

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favouring intervals of 1-100, with evidence weight growing with set size, but as the session wears on they increasingly answer "yes" without consulting their concept at all — a blind-"yes" lapse whose rate rises steadily with how far into the session the trial falls (an inferred starting rate and growth rate). The one change is this session-growing acquiescence, which predicts more "yes" late in the session than early, chiefly to targets the concept clearly excludes (where a blind "yes" makes the most difference), closing the session-liberalization discrepancy the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/session_growing_acquiescence_concepts.py`

### decade_aligned_range_concepts — rank 5, 30.6 ± 11.9 nats behind the best (2.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2827.1

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but they think about number ranges in round decades, so a range whose edges fall on decade boundaries (such as 20-29 or 30-59) is a priori more plausible than an arbitrarily bounded one by an inferred bonus. The one change is this decade-aligned prior bonus on intervals, which predicts that a target just outside the examples' span is endorsed more when it lies in the same decade as an example (e.g. 29 after 23 and 25) than an equally near target across a decade boundary (e.g. 19), the decade-grouping discrepancy the critique of the best model flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/decade_aligned_range_concepts.py`

### log_compressed_sampling_concepts — rank 6, 32.6 ± 13.8 nats behind the best (2.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2829.1

**Hypothesis:** People do the same conservative Bayesian concept learning over arithmetic rules and number ranges as the best model (size principle, evidence weight growing with set size, decade-aligned ranges favoured), but they imagine the program's examples as drawn evenly along a compressed, roughly logarithmic mental number line rather than evenly over the integers, so a concept's small members are expected far more often than its large ones. The one change is this inferred compression of the sampling assumption, which predicts that a large example is weak evidence for a narrow range around it (generalization from 85 spreads much wider in absolute terms than from 5), and that a concept whose members are mostly large is not penalised for its size as much as one spanning small numbers.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/log_compressed_sampling_concepts.py`

### prior_informed_guessing_concepts — rank 7, 42.3 ± 11.1 nats behind the best (3.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2838.9

**Hypothesis:** Refinement of `evidence_scarce_guessing_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100 (evidence weight growing with set size), and they likewise often stop using the examples when there are few of them, less so as more are shown — but when they do, they do not guess "yes" at one blind rate for every target; they answer from their prior beliefs about what such programs produce, as if no examples had been shown. The one change is this prior-informed guessing, which predicts that guessed "yes" answers (most common after a single example) go mainly to targets that many a-priori plausible concepts contain — mid-range and even numbers — and rarely to targets few concepts contain (e.g. 1, 97, 100, odd non-primes near the ends), rather than equally to every non-member.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/prior_informed_guessing_concepts.py`

### log_compressed_range_concepts — rank 8, 45.1 ± 12.7 nats behind the best (3.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2841.6

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning (42 arithmetic rules plus midpoint-weighted, decade-favoured intervals, evidence weight growing with set size), but they judge how big a number range is on a compressed mental number line, where the distance from 80 to 90 feels smaller than the distance from 5 to 15. The one change is that the size principle measures an interval's size, and how likely it is to have produced each example, on this compressed scale (an inferred compression exponent; no compression recovers the incumbent), so generalization is scale-invariant: people extend a range around large examples (e.g. 85, 88) much further in absolute terms than around small ones (e.g. 5, 8) — the region, single and clustered examples at the two ends of 1-100, where it disagrees most with the incumbent's uniform-width generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/log_compressed_range_concepts.py`

### decade_concepts_lapsing_learner — rank 9, 70.6 ± 13.3 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2867.2

**Hypothesis:** Refinement of `lapsing_growing_evidence_concepts`: people do the same conservative Bayesian concept learning (42 arithmetic rules plus midpoint-weighted intervals, evidence weight growing with set size, blind "yes"/"no" lapses), but they also treat the decades of the written number system (1-9, 10-19, ..., 90-99) as privileged, ready-made concepts with their own inferred prior share, over and above the same ranges' standing as ordinary intervals. The one change is this decade hypothesis family, which predicts that a nearby target outside the examples' range is endorsed more when it shares the examples' tens digit than when it crosses into the next decade — the same-decade bonus the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/decade_concepts_lapsing_learner.py`

### session_drifting_decade_concepts — rank 10, 72.1 ± 13.5 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2868.6

**Hypothesis:** Refinement of `decade_concept_bayesian_learning`: people do the same conservative Bayesian concept learning over arithmetic rules, midpoint-weighted intervals and decade concepts (evidence weight growing with set size, blind "yes"/"no" lapses), but as the long session wears on they increasingly answer "yes" without consulting their concept, so the chance of a blind "yes" drifts with how far into the session the trial falls, at an inferred rate. The one change is this session-drifting yes-lapse rate, which predicts that targets the concept excludes are endorsed more often late in the session than early — the growing liberality the critique of the best model flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/session_drifting_decade_concepts.py`

### decade_concept_bayesian_learning — rank 11, 73.0 ± 13.5 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2869.6

**Hypothesis:** People do the same conservative Bayesian concept learning as the best model (42 arithmetic rules and midpoint-weighted intervals of 1-100, evidence weight growing with set size, attention lapses), but they also hold decades — "numbers in the 20s", "numbers in the 60s" (same tens digit) — as named concepts with their own inferred prior share, split equally across the ten decades. Because a decade is a privileged, a-priori-likely range rather than one of thousands of arbitrary intervals, a target just outside the examples' range is endorsed more when it shares their tens decade than when an equally close target crosses a decade boundary.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/decade_concept_bayesian_learning.py`

### tversky_ratio_feature_similarity — rank 12, 129.0 ± 35.5 nats behind the best (3.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2925.6

**Hypothesis:** People judge the target by its feature-based similarity to each example separately, in the sense of Tversky's ratio model: a number is a bundle of salient properties (odd/even, final digit, decade, digits it contains, prime, square, cube, power-of-k, multiple-of-k), and the target's similarity to an example is the share of their combined features they have in common, discounted more for the example's features the target lacks than for the target's own extra features (both discount weights inferred). The higher the target's average similarity to the examples, the more likely people say "yes" — so a target is rejected chiefly for lacking properties the examples have (e.g. the wrong parity), whatever its distance on the number line.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/tversky_ratio_feature_similarity.py`

### fatigue_driven_guessing_concepts — no comparison row

**Hypothesis:** People do the same conservative Bayesian concept learning as the best model (42 arithmetic rules and midpoint-weighted, decade-favoured number ranges, evidence weight growing with set size, an inferred tolerance for stray examples), but the trials on which they skip their concept and simply guess (with their own leaning toward "yes") come from fatigue, not from a shortage of examples: the chance of guessing rises with how far into the session the trial falls, at an inferred rate, and does not depend on how many examples were shown. This predicts that excluded targets get more "yes" late in the session than early (the late-minus-early drift the best model misses), and — unlike the best model — that guessed "yes" answers to non-members are as common after four examples as after one at the same point in the session.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/fatigue_driven_guessing_concepts.py`

### primacy_anchored_exception_concepts — no comparison row

**Hypothesis:** Refinement of `outlier_tolerant_rule_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100 (evidence weight growing with set size, an inferred tolerance for stray examples, guessing that becomes rarer as more examples are shown), but they read the examples in the order shown and take the first one as their anchor: the first example is always trusted as a genuine output of the program's rule, and only examples arriving after it can be dismissed as stray exceptions. The one change is this primacy-anchored exception tolerance, which predicts that a set whose odd-one-out comes later (e.g. 92, 82, 27, 42) keeps the rule the others share, whereas a set whose odd-one-out is the first example (e.g. 27, 92, 82, 42) forces every surviving concept to include that first number, so far targets fitting only the later examples' rule are rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/primacy_anchored_exception_concepts.py`

### span_relative_powerlaw_similarity — no comparison row

**Hypothesis:** People judge a target purely by its closeness on the number line to the span the examples cover: any target inside that span is judged fully similar, and a target outside it loses similarity by a power law in its distance beyond the nearest edge measured relative to the span's own width (plus an inferred baseline width, so a lone example still generalizes to its neighbours). The falloff grows steeper with every additional example, so a few spread-out examples license wide generalization while many tightly clustered ones license almost none; apart from occasional blind guesses, "yes" follows this relative-distance similarity.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/span_relative_powerlaw_similarity.py`

## Pruned models (out of the set; narrowest margin first)

### criterion_decision_decade_concepts — pruned (experiment1 end of experiment)

**Margin:** 6.6 nats behind session_growing_acquiescence_concepts (2.0× dse)

**Hypothesis:** Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100, with evidence weight growing with set size, but they do not report their belief by probability matching — they say "yes" when their confidence that the target belongs clears a personal criterion, applied noisily (the decision rule grafted from `soft_criterion_concept_decision`). The one change is this criterion-based decision stage (an inferred criterion and decision sharpness), which predicts that targets the learner believes moderately are pushed toward a consistent answer, and that near-certain inclusions and exclusions are softened, rather than "yes" rates equal to the posterior probability everywhere.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/criterion_decision_decade_concepts.py`

### decade_aligned_interval_concepts — pruned (experiment1 end of experiment)

**Margin:** 10.2 nats behind session_growing_acquiescence_concepts (3.3× dse)

**Hypothesis:** Refinement of `lapsing_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size and attention lapses, but they mentally carve the number line into decades, so number ranges whose edges fall on decade boundaries (e.g. 20-29, 30-59, 1-10) are a priori more plausible than arbitrary ranges by an inferred factor. The one change is this decade-aligned prior bonus on intervals, which predicts that a nearby target outside the examples' span is endorsed more often when it shares a decade with an example (25 after 21, 23) than when it lies just across a decade boundary (31 after 27, 29) — the same-decade bonus the incumbent's critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/decade_aligned_interval_concepts.py`

### lapsing_growing_evidence_concepts — pruned (experiment1 end of experiment)

**Margin:** 11.0 nats behind session_growing_acquiescence_concepts (2.5× dse)

**Hypothesis:** Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but on some trials they answer without consulting their concept at all, guessing "yes" or "no" with their own inferred rates. The one change is this attention-lapse response floor and ceiling (an inferred chance of a blind "yes" and a separate chance of a blind "no"), which predicts a steady baseline of "yes" to targets the concept clearly excludes and occasional "no" to targets it clearly contains, where the incumbent puts both at near-certainty.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/lapsing_growing_evidence_concepts.py`

### sample_size_growing_evidence_concepts — pruned (experiment1 end of experiment)

**Margin:** 15.8 nats behind session_growing_acquiescence_concepts (3.1× dse)

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/sample_size_growing_evidence_concepts.py`

### soft_criterion_concept_decision — pruned (experiment1 end of experiment)

**Margin:** 22.5 nats behind session_growing_acquiescence_concepts (2.0× dse)

**Hypothesis:** People infer the hidden concept by plain Bayesian concept learning over arithmetic rules and numeric intervals (the simplest current inference, with the size principle), but they do not report that belief by probability matching: they answer "yes" when their confidence that the target belongs clears a personal criterion, applied noisily. How high the criterion sits (a bias toward "yes" or "no") and how sharply confidence is turned into a choice are the only changes, so moderately believed targets are pushed toward one answer and near-certain beliefs are softened toward the other.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/soft_criterion_concept_decision.py`

### iter1_candidate4 — pruned (experiment1 end of experiment)

**Margin:** 23.1 nats behind session_growing_acquiescence_concepts (3.0× dse)

**Hypothesis:** Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and every interval of 1-100 (inferred rule weights, a Beta midpoint prior on intervals, size-principle sampling with 10% noise, examples' evidence weighted by an inferred strength), but they also hold a prior preference for intervals of intermediate size (Tenenbaum's Erlang size prior, scale inferred, grafted from `bp_rules_erlang_interval_size`) that multiplies the midpoint prior. The one change is this size prior, which discounts tiny near-singleton intervals a priori and so predicts broader generalization from a single example — where the incumbent says "yes" too rarely — while leaving multi-example sets, whose evidence already rules out tiny intervals, largely unchanged.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/iter1_candidate4.py`

### first_example_weak_erlang_concepts — pruned (experiment1 end of experiment)

**Margin:** 26.0 nats behind session_growing_acquiescence_concepts (2.7× dse)

**Hypothesis:** Refinement of `first_example_weak_sampling_concepts`: people update their concept one example at a time, and the first example only rules out concepts that do not contain it, while every later example is treated as drawn from the concept (size principle, weighted by an inferred strength) — but people also hold a prior preference for number ranges of intermediate size (Tenenbaum's Erlang size prior on intervals, scale inferred, grafted from `iter1_candidate4`). The one change is this size prior: because a single example carries no size-principle evidence under this account, what people generalize to from one example is set entirely by their prior, and the size prior predicts a moderately wide range around a lone example rather than the broad ranges near 1-100 that a size-blind prior favours.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/first_example_weak_erlang_concepts.py`

### first_example_weak_sampling_concepts — pruned (experiment1 end of experiment)

**Margin:** 28.8 nats behind session_growing_acquiescence_concepts (2.8× dse)

**Hypothesis:** People update their concept one example at a time, and the size principle only kicks in once there is a coincidence to explain: the first example merely rules out concepts that do not contain it (any consistent concept stays as plausible as its prior says), while every later example is treated as drawn from the concept, so small concepts that keep fitting become favoured ("suspicious coincidence"). This predicts broad generalization from a single example and a sharp tightening only when a second consistent example arrives.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/first_example_weak_sampling_concepts.py`

### conservative_evidence_interval_concepts — pruned (experiment1 end of experiment)

**Margin:** 30.9 nats behind session_growing_acquiescence_concepts (3.5× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_interval`: people do the same Bayesian concept learning over 42 arithmetic rules and every interval of 1-100 (a prior weight inferred for each rule, an interval prior given by a Beta density over its midpoint, examples sampled from the concept by the size principle with 10% noise), but they are conservative updaters: they weight the evidence of the examples by a single inferred strength rather than at full Bayesian value, so the posterior over concepts is flatter (strength below 1) or sharper (above 1) than the ideal learner's. The one change is this evidence-weighting exponent on the examples' likelihood, which governs how quickly generalization tightens around the narrowest consistent concept as examples accumulate.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/conservative_evidence_interval_concepts.py`

### win_stay_lose_sample_concepts — pruned (experiment1 end of experiment)

**Margin:** 34.7 nats behind session_growing_acquiescence_concepts (3.9× dse)

**Hypothesis:** People do not keep a full belief over every concept; they read the examples one at a time while holding a single working guess (win-stay, lose-sample). After the first example they pick a concept in proportion to how well it explains that example; each later example that the guess contains leaves it in place, and only an example it fails to contain makes them drop it and draw a new guess from what all the examples seen so far support. They answer "yes" if their current guess contains the target (apart from occasional blind guesses), so consistent later examples do not narrow the guess the way an ideal learner's belief narrows, and the order of the examples changes which concept people end up holding.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/win_stay_lose_sample_concepts.py`

### recency_decayed_evidence_concepts — pruned (experiment1 end of experiment)

**Margin:** 35.9 nats behind session_growing_acquiescence_concepts (4.0× dse)

**Hypothesis:** People learn the program's concept by conservative Bayesian concept learning over arithmetic rules and numeric intervals (size principle), but their memory of the examples fades with how long ago each was shown: the last example counts as full evidence, and each example further back in the sequence counts as progressively weaker evidence, by a constant inferred retention factor per position. Generalization is therefore driven mostly by the most recent examples, so reordering the same examples changes which targets get a "yes" (targets that fit only the first example are endorsed less often than targets that fit only the last).

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/recency_decayed_evidence_concepts.py`

### size_principle_strength_intervals — pruned (experiment1 end of experiment)

**Margin:** 57.8 nats behind session_growing_acquiescence_concepts (5.0× dse)

**Hypothesis:** Refinement of bigelow_piantadosi_interval: people use the same Bayesian concept learning over the 42 rules and the midpoint-weighted intervals of 1-100, but they apply the size principle only partially — how strongly a small concept is favoured for fitting the examples tightly is a graded, inferred strength, between weak sampling (examples merely consistent with a concept, no preference for small concepts) and the full strong sampling the incumbent assumes. The one change is this inferred size-principle strength, which predicts broader generalization than the incumbent (e.g. more "yes" to other even numbers after 16, 8, 64) if people underweight concept size.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/size_principle_strength_intervals.py`

### digit_place_similarity_gradient — pruned (experiment1 end of experiment)

**Margin:** 59.8 nats behind session_growing_acquiescence_concepts (3.0× dse)

**Hypothesis:** People judge a target by its similarity to the examples in their written decimal form rather than on the number line: they compare the target with each example place by place, and every mismatch in the units digit or in the tens digit multiplies similarity down by its own inferred factor. The more similar the target is on average to the examples under this digit-place metric, the more likely people are to say "yes" — so 21 after 91, or 58 after 28 and 98, is endorsed despite being numerically far away, while 88 after 89 is rejected despite being adjacent.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/digit_place_similarity_gradient.py`

### shared_property_overlap — pruned (experiment1 end of experiment)

**Margin:** 68.7 nats behind session_growing_acquiescence_concepts (4.1× dse)

**Hypothesis:** People do not weigh concepts by how tightly they fit the examples (no size principle); instead they note every salient number property that all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of those shared properties the target also has, with every shared property counting equally. This disagrees most sharply with the best model on targets that share the examples' broad properties but not their narrowest one (e.g. 14 after 28, 98: even but not ending in 8), which this account rates as half-matching while the size-principle model gives them almost no generalization.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/shared_property_overlap.py`

### exemplar_feature_similarity_gcm — pruned (experiment1 end of experiment)

**Margin:** 77.4 nats behind session_growing_acquiescence_concepts (3.9× dse)

**Hypothesis:** People do not induce a rule; they judge the target by its similarity to the stored examples (exemplar generalization, as in the Generalized Context Model). Each number is represented by its magnitude and its salient arithmetic/digit properties (parity, primality, square, cube, multiples, last and tens digit), and the more similar the target is on average to the examples — similarity decaying exponentially with a weighted distance in this space — the more likely people are to say yes, with no hypothesis space, posterior, or size principle involved.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/exemplar_feature_similarity_gcm.py`

### coherence_attention_exemplar_similarity — pruned (experiment1 end of experiment)

**Margin:** 101.5 nats behind session_growing_acquiescence_concepts (4.8× dse)

**Hypothesis:** People do not infer a concept; they judge the target by its similarity to the stored examples (exemplar generalization), but their attention is drawn to the dimensions on which the stored examples cohere: every property all the examples share (parity, final digit, decade, contained digit, prime, square, cube, power-of-k, multiple-of-k) gets attention in proportion to how distinctive it is in 1-100, so a target that lacks a rare shared property (e.g. "ends in 8", "in the 20s") is judged dissimilar to every example, while closeness on the number line counts on a scale that stretches with the examples' spread. Saying "yes" is a Luce choice between the target's mean similarity to the examples and a fixed criterion.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/coherence_attention_exemplar_similarity.py`

### transformational_distance_similarity — pruned (experiment1 end of experiment)

**Margin:** 134.7 nats behind session_growing_acquiescence_concepts (6.0× dse)

**Hypothesis:** People do not induce a concept or measure closeness on the number line; they judge whether the target can be made from one of the examples by a short chain of simple mental operations on numbers (add or subtract one, add or subtract ten, double, halve, reverse the digits), so the fewer operations it takes to turn the examples into the target, the more likely they say "yes" (transformational similarity). This predicts "yes" to 32 after 16, to 61 after 16, and to 40 after 20 even though they are far away on the number line, while a numerically moderate jump that needs many steps (e.g. 47 after 20) is rejected.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/transformational_distance_similarity.py`

### range_scaled_magnitude_gradient — pruned (experiment1 end of experiment)

**Margin:** 135.1 nats behind session_growing_acquiescence_concepts (6.0× dse)

**Hypothesis:** People do not consider arithmetic rules or properties at all; they judge a target purely by its closeness on the number line to the nearest example, with a generalization gradient whose width stretches with the spread of the examples. A single example (no spread) still supports a fixed baseline width of generalization around it, so "yes" falls off smoothly with numerical distance, more slowly when the examples cover a wide range.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/range_scaled_magnitude_gradient.py`

### bp_rules_erlang_interval_size — pruned (experiment1 end of experiment)

**Margin:** 186.9 nats behind session_growing_acquiescence_concepts (6.3× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 42 rules (each with its own inferred prior weight) and every interval of 1-100, with examples sampled from the concept (the size principle, 10% noise), and answer "yes" with the posterior probability that the concept contains the target. The one change is that intervals are not equally likely a priori: people prefer intervals of intermediate size, with an Erlang prior on an interval's size whose scale is inferred (Tenenbaum's size prior grafted onto the independent-rule-weight model), so a few nearby examples generalize to a moderately wide range rather than to the very tightest or the very widest interval.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bp_rules_erlang_interval_size.py`

### bigelow_piantadosi_interval — pruned (experiment1 end of experiment)

**Margin:** 206.5 nats behind session_growing_acquiescence_concepts (6.9× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group. An interval's prior follows a Beta density over its midpoint, with both shapes inferred, so concepts in some parts of 1-100 are favoured. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_interval.py`

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Margin:** 212.5 nats behind session_growing_acquiescence_concepts (6.5× dse)

**Hypothesis:** Bayesian concept learning over 42 rules (powers of 2-10, multiples of 3-12, final digit 1-9, contained digit 1-9, primes, even, odd, squares, cubes) and every interval of 1-100, with a prior weight inferred for each rule separately and a share p_mix for the rules as a group; intervals are equally likely. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/bigelow_piantadosi_independent.py`

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Margin:** 299.4 nats behind session_growing_acquiescence_concepts (6.5× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

**Source:** `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/pruned/tenenbaum_rules_similarity.py`
