# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, best first, then models pruned earlier, closest to the best first (a pruned mechanism may come back with a substantive change).

## rsa_l2 (best on exp1 (held out, grouped CV); 250.2 ± 62.9 behind on E; cross-validation folds did not converge, so it is untrusted: fold 2: 993 divergent transitions of 2000; max R-hat 2.483 > 1.05; min bulk ESS 5 < 100)

RSA pragmatic listener at depth 2, uniform prior over objects. One more level of recursion than the vanilla model: a speaker S2 chooses words by how well the depth-1 pragmatic listener L1 would recover the object (exp(alpha * log L1)), and the listener L2 inverts S2. Depth-2 reasoning is what the pragmods "complex" game needs to single out the object described by the twice-implicated word. One rationality alpha at every level; uniform choice with no informative word; a lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/510e4bc2d15e51db/repo/_runs/chain_0/experiment1/loop/models/rsa_l2.py`

## isolated_graded_costly_l3 (137988.0 ± 39775.0 nats behind the best on exp1 (held out, grouped CV); 145.0 ± 57.6 behind on E)

Pragmatic listener at depth 3 with perceptual isolation, graded semantics, and utterance extension costs. Refines isolated_graded_costly_l2 by increasing the recursion depth from depth 2 to depth 3. Listeners reason at depth 3, inverting a speaker who simulates a depth-2 pragmatic listener. Visual pop-out is determined by an object's perceptual isolation (minimum Hamming distance to any other context object), semantic applicability diminishes with additional competing features (graded truth), and speakers evaluate communicative utility trade-offs with utterance extension costs proportional to the fraction of display objects sharing each feature. On uninformative prior trials, choice follows the perceptual isolation prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/510e4bc2d15e51db/repo/_runs/chain_0/experiment1/loop/models/isolated_graded_costly_l3.py`

## competitor_confusion_speaker (138378.8 ± 39769.9 nats behind the best on exp1 (held out, grouped CV); 195.2 ± 56.4 behind on E)

Pragmatic listener reasoning about a competitor-confusion-averse speaker. Speakers in reference games actively avoid referring expressions that are shared with visually similar or confusable distractors in the context. Candidate words are penalized proportionally to their feature overlap with alternative referents that also satisfy the word. Pragmatic listeners invert this confusion-averse speaker, expecting words to target referents whose competitors are visually distinct rather than near-identical clones.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/510e4bc2d15e51db/repo/_runs/chain_0/experiment1/loop/models/competitor_confusion_speaker.py`

## oddity_heuristic_listener (a cross-validation fold failed, so it is unscored: oddity_heuristic_listener: the fit did not finish within 28 minutes (it was still in warmup, adapting its step size: the posterior is hard to explore))

Perceptual oddity heuristic listener. Listeners interpret referring expressions using a perceptual oddity heuristic rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners select the contextually unique singleton that lacks identical duplicates in the visual display. On uninformative trials without an informative word, listeners spontaneously choose the unique singleton, defaulting to uniform choice when all items are distinct or no oddity contrast exists. A lapse parameter captures random clicking.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/510e4bc2d15e51db/repo/_runs/chain_0/experiment1/loop/models/oddity_heuristic_listener.py`

## rsa_l1 (a cross-validation fold failed, so it is unscored: rsa_l1: the fit did not finish within 18 minutes (it was still in warmup, adapting its step size: the posterior is hard to explore))

Vanilla RSA pragmatic listener (depth 1), uniform prior over objects. The listener reasons about a speaker who chooses a true word with probability proportional to exp(alpha * log L0(object | word)), where L0 is the literal listener; the listener inverts that speaker with a uniform prior over objects (Frank & Goodman, 2012). With no informative word the choice is uniform. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/510e4bc2d15e51db/repo/_runs/chain_0/experiment1/loop/models/rsa_l1.py`
