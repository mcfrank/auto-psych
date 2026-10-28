# Refinement menu

Choose ONE model to refine (not the incumbent). Your hypothesis must name it and state the single change you make. Live models first, best first, then models pruned earlier, closest to the best first (a pruned mechanism may come back with a substantive change).

## isolated_graded_costly_l3 (335.6 ± 83.0 nats behind the best on exp1 (held out, grouped CV); 146.4 ± 55.8 behind on E)

Pragmatic listener at depth 3 with perceptual isolation, graded semantics, and utterance extension costs. Refines isolated_graded_costly_l2 by increasing the recursion depth from depth 2 to depth 3. Listeners reason at depth 3, inverting a speaker who simulates a depth-2 pragmatic listener. Visual pop-out is determined by an object's perceptual isolation (minimum Hamming distance to any other context object), semantic applicability diminishes with additional competing features (graded truth), and speakers evaluate communicative utility trade-offs with utterance extension costs proportional to the fraction of display objects sharing each feature. On uninformative prior trials, choice follows the perceptual isolation prior. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/isolated_graded_costly_l3.py`

## competitor_confusion_speaker (451.6 ± 126.8 nats behind the best on exp1 (held out, grouped CV); 193.2 ± 58.2 behind on E)

Pragmatic listener reasoning about a competitor-confusion-averse speaker. Speakers in reference games actively avoid referring expressions that are shared with visually similar or confusable distractors in the context. Candidate words are penalized proportionally to their feature overlap with alternative referents that also satisfy the word. Pragmatic listeners invert this confusion-averse speaker, expecting words to target referents whose competitors are visually distinct rather than near-identical clones.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/competitor_confusion_speaker.py`

## rsa_l1 (1232.0 ± 195.5 nats behind the best on exp1 (held out, grouped CV); 250.2 ± 65.6 behind on E; cross-validation folds did not converge, so it is untrusted: fold 3: max R-hat 1.529 > 1.05; min bulk ESS 7 < 100)

Vanilla RSA pragmatic listener (depth 1), uniform prior over objects. The listener reasons about a speaker who chooses a true word with probability proportional to exp(alpha * log L0(object | word)), where L0 is the literal listener; the listener inverts that speaker with a uniform prior over objects (Frank & Goodman, 2012). With no informative word the choice is uniform. A lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/rsa_l1.py`

## rsa_l2 (1340.0 ± 200.5 nats behind the best on exp1 (held out, grouped CV); 193.6 ± 59.8 behind on E)

RSA pragmatic listener at depth 2, uniform prior over objects. One more level of recursion than the vanilla model: a speaker S2 chooses words by how well the depth-1 pragmatic listener L1 would recover the object (exp(alpha * log L1)), and the listener L2 inverts S2. Depth-2 reasoning is what the pragmods "complex" game needs to single out the object described by the twice-implicated word. One rationality alpha at every level; uniform choice with no informative word; a lapse parameter mixes in uniform guessing.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/rsa_l2.py`

## oddity_heuristic_listener (1526.8 ± 231.2 nats behind the best on exp1 (held out, grouped CV); 704.5 ± 136.0 behind on E; NOT eligible to lead: more than 4 SEs behind the best on E)

Perceptual oddity heuristic listener. Listeners interpret referring expressions using a perceptual oddity heuristic rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners select the contextually unique singleton that lacks identical duplicates in the visual display. On uninformative trials without an informative word, listeners spontaneously choose the unique singleton, defaulting to uniform choice when all items are distinct or no oddity contrast exists. A lapse parameter captures random clicking.

Source: `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/oddity_heuristic_listener.py`
