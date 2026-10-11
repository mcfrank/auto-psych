We refine competitor_confusion_speaker by extending the pragmatic listener to integrate the speaker's confusion-based object prior into utterance interpretation, so that listeners expect referring expressions to target referents with visually distinct competitors rather than assuming referents are chosen uniformly at random. This allows competitor distinctiveness to guide referent resolution on both word and prior trials, resolving the discrepancy where singleton referents were previously under-selected upon hearing shared utterances.

Differences from competitor_confusion_speaker:
- Recursion depth: Unchanged at depth 1 (choice_probs calls L1).
- Parameters: None added or removed (retains alpha, w_confusion, and lapse).
- Other terms: In L1, the simulated speaker's referent distribution is changed from uniform (wpp=1) to the contextual confusion prior (wpp=vec(prior, r)), which choice_probs passes as prior: ... to L1.
