# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, best first, then models pruned earlier, closest to the best first (a pruned mechanism may come back with a substantive change).

## bounded_depth_listener (0.1 ± 20.3 nats behind the best on exp2 (held out, grouped CV); 166.9 ± 57.9 behind on E, 8.6 ± 9.8 behind on exp1)

Listeners interpret referring expressions with a strictly bounded reasoning depth, inverting a communicative speaker who reasons about literal semantics rather than engaging in higher-order recursive mental simulation. This bounded depth prevents over-implicature, naturally preserving non-zero probability for multi-feature scalar competitors while strongly favoring minimal referents when words are universally shared. On uninformative displays without an informative word, visual attention defaults to contextually unique singletons.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/models/bounded_depth_listener.py`

## rsa_l2_singleton_feat_color_valence_l0 (9.1 ± 62.5 nats behind the best on exp2 (held out, grouped CV); best on E, 29.4 ± 21.3 behind on exp1)

RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_distinct_feat_color_l0 by incorporating the evaluative valence framing component from polarity_congruence_listener: speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects with no duplicates in the scene), continuous contextual distinctiveness (favoring objects with greater feature contrast relative to the rest of the display), visual feature complexity modulated by evaluative framing valence (favoring feature-rich items under neutral/positive framing but inverting to favor sparse, featureless items when referring to a least favorite object), and visual color contrast (favoring fully colored objects over desaturated grayscale distractors). Crucially, the foundational simulated literal listener L0 operates purely on truth-conditional semantics without perceptual bias, preventing redundant prior compounding across reasoning tiers. On prior trials with no informative word, choices are governed directly by the combined perceptual prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/models/rsa_l2_singleton_feat_color_valence_l0.py`

## bounded_depth_feature_parsimony (9.8 ± 23.9 nats behind the best on exp2 (held out, grouped CV); 160.1 ± 57.3 behind on E, 10.0 ± 10.2 behind on exp1)

Listeners interpret referring expressions with strictly bounded reasoning depth (depth 1) by inverting an informative speaker over literal semantics, but penalize candidate referents possessing redundant visual features using a feature parsimony heuristic taken from feature_parsimony_listener. On informative trials, this parsimony penalty prevents visual singleton salience from spuriously overriding communicative informativeness for multi-feature competitors, while on uninformative displays visual attention defaults to contextually unique singletons. Differences from bounded_depth_listener: recursion depth remains depth 1 (listener calls L1); parameter beta is added with prior dist.LogNormal(0.0, 1.0); no parameters are removed (retaining alpha and w_singleton); L1 adds feature_count to its signature and scales referent choice weights by exp(-beta * vec(feature_count, r)); and listener passes beta and ctx.feature_count into L1 on informative trials.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/models/bounded_depth_feature_parsimony.py`

## topic_discriminability_listener (admitted this round, not yet scored)

Listeners interpret referring expressions by modeling what speakers are likely to talk about before an utterance is heard: a referential prior over objects grounded in communicative discriminability. Before hearing a word, listeners expect speakers to choose topics that can be successfully distinguished in context, assigning prior probability to each candidate referent according to the maximal literal communicative recovery of its features. On uninformative displays or universal utterances, this topic prior directs belief toward uniquely discriminable objects without relying on spurious visual singleton heuristics that distort scalar implicatures.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/models/topic_discriminability_listener.py`
