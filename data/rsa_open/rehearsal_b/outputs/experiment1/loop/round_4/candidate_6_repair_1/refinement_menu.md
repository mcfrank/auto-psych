# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, best first, then models pruned earlier, closest to the best first (a pruned mechanism may come back with a substantive change).

## feature_parsimony_listener (best on exp1 (held out, grouped CV); 478.0 ± 99.8 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Listeners interpret referring expressions using a feature parsimony heuristic: among candidate referents satisfying a heard word, listeners choose the object possessing the fewest total visual features, assuming speakers provide minimal descriptions for simple targets rather than leaving distinguishing features unmentioned. Speakers communicate by selecting words that guide this parsimony-biased listener to the intended referent. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/feature_parsimony_listener.py`

## rsa_l1_singleton_feat_color_valence_l0 (117347.7 ± 54038.1 nats behind the best on exp1 (held out, grouped CV); 61.3 ± 23.7 behind on E)

RSA pragmatic listener at depth 1 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_feat_color_valence_l0 by reducing pragmatic recursion depth from depth 2 to depth 1: listeners interpret referring expressions by inverting an informative speaker who evaluates literal semantics, rather than engaging in higher-order recursive mental simulation. This bounded depth eliminates over-implicature in scalar implicature and prevents multiplicative prior compounding across reasoning tiers, preserving appropriate probability on multi-feature competitors while maintaining common knowledge of visual singleton salience, continuous contextual distinctiveness, valence-modulated feature complexity, and color contrast. Differences from rsa_l2_singleton_feat_color_valence_l0: recursion depth is reduced from depth 2 to depth 1 (listener calls L1 instead of L2), the depth-2 listener memo L2 is removed, and all parameters (alpha, w_singleton, w_distinct, w_features, w_valence, w_color) are preserved without addition or removal.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/rsa_l1_singleton_feat_color_valence_l0.py`

## rsa_l2_singleton_feat_color_valence_l0 (117384.8 ± 54031.8 nats behind the best on exp1 (held out, grouped CV); best on E)

RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_distinct_feat_color_l0 by incorporating the evaluative valence framing component from polarity_congruence_listener: speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects with no duplicates in the scene), continuous contextual distinctiveness (favoring objects with greater feature contrast relative to the rest of the display), visual feature complexity modulated by evaluative framing valence (favoring feature-rich items under neutral/positive framing but inverting to favor sparse, featureless items when referring to a least favorite object), and visual color contrast (favoring fully colored objects over desaturated grayscale distractors). Crucially, the foundational simulated literal listener L0 operates purely on truth-conditional semantics without perceptual bias, preventing redundant prior compounding across reasoning tiers. On prior trials with no informative word, choices are governed directly by the combined perceptual prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/rsa_l2_singleton_feat_color_valence_l0.py`

## structured_lapse_listener (117392.2 ± 54060.6 nats behind the best on exp1 (held out, grouped CV); 440.7 ± 96.9 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

People separate communicative inference from response generation: listeners infer communicative intent using an unbiased, uniform target prior, and resolve their response through a salience-guided structured lapse that captures attention toward contextually unique visual singletons. On uninformative displays lacking a communicative word, choices default entirely to visual singleton salience; on informative trials, communicative beliefs are mixed at the decision level with the salience lapse rather than multiplied inside Bayes' rule.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/structured_lapse_listener.py`

## grammatical_exhaustification (117488.2 ± 54033.5 nats behind the best on exp1 (held out, grouped CV); 364.2 ± 77.1 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

People resolve referring expressions using grammatical exhaustification over unasserted alternative descriptions rather than recursive Theory of Mind. When hearing a word, listeners rule out referents proportionally to the informativeness of any truer or more specific features of that referent that the speaker left unmentioned. On uninformative trials, visual attention defaults to contextually unique singletons.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/grammatical_exhaustification.py`

## grammatical_exhaustification_distinct (117493.7 ± 54040.4 nats behind the best on exp1 (held out, grouped CV); 229.7 ± 60.9 behind on E)

Refining grammatical_exhaustification by incorporating continuous contextual distinctiveness into the visual attention prior: listeners interpret referring expressions through grammatical exhaustification over unasserted alternative features, ruling out referents proportionally to the informativeness of any more specific descriptions the speaker omitted, while spontaneous visual attention weights referents by combining discrete singleton salience with continuous feature contrast (contextual distinctiveness) relative to display competitors. This directly improves the representation of displays lacking identical duplicates as well as empty singletons in full-duplicate contexts, addressing the empirical discrepancy where humans allocate substantial prior belief to empty singleton objects among identical full-feature duplicates. Differences from grammatical_exhaustification: recursion depth remains unchanged (depth 0, L_exh); parameter w_distinct is added with prior dist.Normal(0.0, 1.0); no parameters are removed; and the visual prior is extended by adding w_distinct times the total contextual feature contrast (summed Hamming distance across all context objects).

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/grammatical_exhaustification_distinct.py`

## isolated_graded_costly_l3 (117668.7 ± 54036.7 nats behind the best on exp1 (held out, grouped CV); 419.6 ± 100.3 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Pragmatic listener at depth 3 with perceptual isolation, graded semantics, and utterance extension costs. Refines isolated_graded_costly_l2 by increasing the recursion depth from depth 2 to depth 3. Listeners reason at depth 3, inverting a speaker who simulates a depth-2 pragmatic listener. Visual pop-out is determined by an object's perceptual isolation (minimum Hamming distance to any other context object), semantic applicability diminishes with additional competing features (graded truth), and speakers evaluate communicative utility trade-offs with utterance extension costs proportional to the fraction of display objects sharing each feature. On uninformative prior trials, choice follows the perceptual isolation prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/isolated_graded_costly_l3.py`

## bounded_depth_unbiased_prior (117916.8 ± 54103.1 nats behind the best on exp1 (held out, grouped CV); 403.3 ± 105.9 behind on E)

Refining bounded_depth_listener by decoupling visual singleton salience from communicative inference: listeners interpret informative referring expressions with bounded reasoning depth under an unbiased, uniform referent prior, confining visual singleton salience strictly to spontaneous visual attention on uninformative displays without an informative word. This prevents spurious singleton bonuses from overriding scalar implicatures on informative trials, eliminating severe over-allocation to multi-feature singleton competitors when words are shared with duplicate minimal targets. Differences from bounded_depth_listener: recursion depth remains unchanged (depth 1, listener calls L1); no parameters are added or removed (retaining alpha and w_singleton); the prior vector parameter is removed from L1 so that the listener chooses referents proportionally to speaker choice probabilities alone; and listener calls L1 without the prior argument, restricting the singleton prior exclusively to prior trials.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/bounded_depth_unbiased_prior.py`

## competitor_confusion_speaker (117979.6 ± 54109.6 nats behind the best on exp1 (held out, grouped CV); 326.8 ± 98.1 behind on E)

Pragmatic listener reasoning about a competitor-confusion-averse speaker. Speakers in reference games actively avoid referring expressions that are shared with visually similar or confusable distractors in the context. Candidate words are penalized proportionally to their feature overlap with alternative referents that also satisfy the word. Pragmatic listeners invert this confusion-averse speaker, expecting words to target referents whose competitors are visually distinct rather than near-identical clones.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/competitor_confusion_speaker.py`

## contextual_oddity_listener (118005.5 ± 54102.9 nats behind the best on exp1 (held out, grouped CV); 174.7 ± 61.3 behind on E)

People interpret referring expressions by integrating communicative informativeness with contextual oddity salience. Before hearing an utterance, visual attention is captured by a solitary odd-one-out that breaks the homogeneous background of duplicate distractors, establishing a default referential prior over objects. Pragmatic listeners invert an informative speaker using this oddity-grounded referent prior, defaulting to uniform expectations when multiple distinct items or no oddities exist.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/contextual_oddity_listener.py`

## oddity_heuristic_listener (118197.4 ± 54059.4 nats behind the best on exp1 (held out, grouped CV); 1773.0 ± 255.1 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Perceptual oddity heuristic listener. Listeners interpret referring expressions using a perceptual oddity heuristic rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners select the contextually unique singleton that lacks identical duplicates in the visual display. On uninformative trials without an informative word, listeners spontaneously choose the unique singleton, defaulting to uniform choice when all items are distinct or no oddity contrast exists. A lapse parameter captures random clicking.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/oddity_heuristic_listener.py`

## rsa_l1 (118355.3 ± 54082.8 nats behind the best on exp1 (held out, grouped CV); 620.5 ± 152.0 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Vanilla RSA pragmatic listener (depth 1), uniform prior over objects. The listener reasons about a speaker who chooses a true word with probability proportional to exp(alpha * log L0(object | word)), where L0 is the literal listener; the listener inverts that speaker with a uniform prior over objects (Frank & Goodman, 2012). With no informative word the choice is uniform. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/rsa_l1.py`

## rsa_l2 (118377.4 ± 54083.0 nats behind the best on exp1 (held out, grouped CV); 570.8 ± 145.6 behind on E)

RSA pragmatic listener at depth 2, uniform prior over objects. One more level of recursion than the vanilla model: a speaker S2 chooses words by how well the depth-1 pragmatic listener L1 would recover the object (exp(alpha * log L1)), and the listener L2 inverts S2. Depth-2 reasoning is what the pragmods "complex" game needs to single out the object described by the twice-implicated word. One rationality alpha at every level; uniform choice with no informative word; a lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/rsa_l2.py`

## bounded_depth_feature_parsimony (admitted this round, not yet scored)

Listeners interpret referring expressions with strictly bounded reasoning depth (depth 1) by inverting an informative speaker over literal semantics, but penalize candidate referents possessing redundant visual features using a feature parsimony heuristic taken from feature_parsimony_listener. On informative trials, this parsimony penalty prevents visual singleton salience from spuriously overriding communicative informativeness for multi-feature competitors, while on uninformative displays visual attention defaults to contextually unique singletons. Differences from bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); L1 adds feature_count to its signature and scales referent choice weights by exp(-beta * vec(feature_count, r)); and listener passes beta and ctx.feature_count into L1 on informative trials.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/models/bounded_depth_feature_parsimony.py`
