# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, best first, then models pruned earlier, closest to the best first (a pruned mechanism may come back with a substantive change).

## contextual_contrast_distinct_listener (1.9 ± 5.9 nats behind the best on exp1 (held out, grouped CV); 95.2 ± 60.2 behind on E)

We refine contextual_contrast_listener by incorporating continuous contextual distinctiveness into the visual referent prior alongside discrete singleton salience, as established in rsa_l2_singleton_feat_color_valence_l0. Speakers maximize contextual contrast against visually similar competitors and listeners invert this speaker, but referent expectations now weight both discrete singleton status and continuous total feature distance to all context distractors, explaining why fully isolated singletons like empty objects attract greater belief than singletons sharing features with duplicate competitors and elevating singleton competitor belief when hearing shared features. Differences from contextual_contrast_listener: recursion depth is unchanged (calls L_contrast); parameter w_distinct is added with prior Normal(0.0, 1.0); the prior logit adds w_distinct multiplied by the object's sum of Hamming distances to all other objects; no parameters or likelihood terms are removed.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/contextual_contrast_distinct_listener.py`

## contrastive_speaker (72.5 ± 31.6 nats behind the best on exp1 (held out, grouped CV); 199.0 ± 60.8 behind on E)

Speakers choose referring expressions to maximize contrast against contextually similar competitors that lack the target feature, rather than merely minimizing extension size. Pragmatic listeners invert this contrastive speaker, correctly expecting an overspecified competitor to receive the referring expression in displays where that expression uniquely differentiates it from near-identical distractors (such as feature 2 in the 4-object scalar display with duplicate foils, sharply disagreeing with the incumbent model's near-zero prediction). On trials without an informative word, listeners default to visual singleton pop-out.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/contrastive_speaker.py`

## avoidable_ambiguity_speaker (128.4 ± 43.4 nats behind the best on exp1 (held out, grouped CV); 207.7 ± 62.3 behind on E)

When an object possesses a uniquely identifying expression that distinguishes it from all context distractors, a cooperative speaker incurs an ambiguity penalty for choosing a shared, non-identifying descriptor instead. When an object lacks any uniquely identifying expression and cannot be uniquely named, the speaker incurs no penalty and is forced to rely on shared descriptors. The pragmatic listener inverts this speaker, inferring that ambiguous words signal referents that could not be uniquely named, while relying on visual singleton salience on uninformative prior trials.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/avoidable_ambiguity_speaker.py`

## rsa_l2_singleton_feat_color_valence_l0 (142.3 ± 42.3 nats behind the best on exp1 (held out, grouped CV); best on E)

RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_distinct_feat_color_l0 by incorporating the evaluative valence framing component from polarity_congruence_listener: speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects with no duplicates in the scene), continuous contextual distinctiveness (favoring objects with greater feature contrast relative to the rest of the display), visual feature complexity modulated by evaluative framing valence (favoring feature-rich items under neutral/positive framing but inverting to favor sparse, featureless items when referring to a least favorite object), and visual color contrast (favoring fully colored objects over desaturated grayscale distractors). Crucially, the foundational simulated literal listener L0 operates purely on truth-conditional semantics without perceptual bias, preventing redundant prior compounding across reasoning tiers. On prior trials with no informative word, choices are governed directly by the combined perceptual prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/rsa_l2_singleton_feat_color_valence_l0.py`

## rsa_l2_uncompounded_l1 (142.8 ± 32.5 nats behind the best on exp1 (held out, grouped CV); 26.5 ± 15.2 behind on E)

Refining rsa_l2_singleton_feat_color_valence_l0 by removing the perceptual referent prior from the simulated depth-1 pragmatic listener L1, speakers evaluate referring expressions against a listener with a uniform prior over referents, while only the top-level pragmatic listener L2 integrates communicative likelihood with the visual perceptual prior combining singleton salience, contextual distinctiveness, valence-modulated complexity, and color contrast. Eliminating redundant prior compounding across reasoning tiers prevents visual salience from overwhelming communicative informativeness, addressing the over-penalization of overspecified competitors and duplicate targets in scalar contexts while preserving singleton pop-out on uninformative prior trials. Differences from source: recursion depth is unchanged (listener calls L2); no parameters added or removed; L1 no longer takes the prior parameter and assumes a uniform prior over objects; L2 evaluates L1 without passing the prior.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/rsa_l2_uncompounded_l1.py`

## attentional_bottleneck_listener (244.2 ± 38.0 nats behind the best on exp1 (held out, grouped CV); 349.1 ± 88.5 behind on E)

Speakers with limited visual attention choose referring expressions to differentiate the target from its nearest confusable context competitor lacking the feature, rather than exhaustively summing contrast across all distractors. Listeners interpret utterances by inverting this attention-limited speaker, while allocating referent expectations on uninformative displays based on perceptual isolation from surrounding items.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/attentional_bottleneck_listener.py`

## contextual_isolation_listener (290.7 ± 47.1 nats behind the best on exp1 (held out, grouped CV); 219.4 ± 72.0 behind on E)

Listeners interpret referring expressions by inverting a contrastive speaker while expecting speakers to refer to contextually isolated outliers rather than items embedded in dense visual clusters. An object's prior referential salience decreases with its contextual similarity density to other items in the display, favoring distinctive singletons over duplicates and isolated items over near-competitors. On uninformative displays without an informative word, listener referent choices follow this contextual isolation prior directly.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/contextual_isolation_listener.py`

## isolated_graded_costly_l3 (426.2 ± 84.7 nats behind the best on exp1 (held out, grouped CV); 419.6 ± 100.3 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Pragmatic listener at depth 3 with perceptual isolation, graded semantics, and utterance extension costs. Refines isolated_graded_costly_l2 by increasing the recursion depth from depth 2 to depth 3. Listeners reason at depth 3, inverting a speaker who simulates a depth-2 pragmatic listener. Visual pop-out is determined by an object's perceptual isolation (minimum Hamming distance to any other context object), semantic applicability diminishes with additional competing features (graded truth), and speakers evaluate communicative utility trade-offs with utterance extension costs proportional to the fraction of display objects sharing each feature. On uninformative prior trials, choice follows the perceptual isolation prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/isolated_graded_costly_l3.py`

## competitor_confusion_speaker (737.1 ± 133.4 nats behind the best on exp1 (held out, grouped CV); 326.8 ± 98.1 behind on E)

Pragmatic listener reasoning about a competitor-confusion-averse speaker. Speakers in reference games actively avoid referring expressions that are shared with visually similar or confusable distractors in the context. Candidate words are penalized proportionally to their feature overlap with alternative referents that also satisfy the word. Pragmatic listeners invert this confusion-averse speaker, expecting words to target referents whose competitors are visually distinct rather than near-identical clones.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/competitor_confusion_speaker.py`

## oddity_heuristic_listener (954.9 ± 144.8 nats behind the best on exp1 (held out, grouped CV); 1773.0 ± 255.1 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Perceptual oddity heuristic listener. Listeners interpret referring expressions using a perceptual oddity heuristic rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners select the contextually unique singleton that lacks identical duplicates in the visual display. On uninformative trials without an informative word, listeners spontaneously choose the unique singleton, defaulting to uniform choice when all items are distinct or no oddity contrast exists. A lapse parameter captures random clicking.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/oddity_heuristic_listener.py`

## rsa_l1 (1112.8 ± 137.0 nats behind the best on exp1 (held out, grouped CV); 620.5 ± 152.0 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Vanilla RSA pragmatic listener (depth 1), uniform prior over objects. The listener reasons about a speaker who chooses a true word with probability proportional to exp(alpha * log L0(object | word)), where L0 is the literal listener; the listener inverts that speaker with a uniform prior over objects (Frank & Goodman, 2012). With no informative word the choice is uniform. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/rsa_l1.py`

## rsa_l2 (1134.9 ± 139.6 nats behind the best on exp1 (held out, grouped CV); 570.8 ± 145.6 behind on E)

RSA pragmatic listener at depth 2, uniform prior over objects. One more level of recursion than the vanilla model: a speaker S2 chooses words by how well the depth-1 pragmatic listener L1 would recover the object (exp(alpha * log L1)), and the listener L2 inverts S2. Depth-2 reasoning is what the pragmods "complex" game needs to single out the object described by the twice-implicated word. One rationality alpha at every level; uniform choice with no informative word; a lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/rsa_l2.py`
