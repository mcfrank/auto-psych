Listeners evaluate referring expressions by combining a softmax decision rule over pragmatic posterior beliefs with an expectation that speakers evaluate candidate utterances based on communicative specificity minus utterance ambiguity cost. When multiple candidate words apply to an intended referent, speakers incur production costs proportional to this referential ambiguity across competitor objects, and listeners invert this cost-sensitive speaker before softly maximizing over their pragmatic posterior beliefs according to a decision rationality parameter. This fits better because pragmatic listeners penalize broad, ambiguous expressions more realistically than literal informativeness alone dictates, while bounded rationality governs final referent choice.

Refinement of softmax_belief_listener:
- Base model refined: softmax_belief_listener
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from softmax_belief_listener).
- Parameters added: cost_ambiguity (prior: Normal(0.0, 1.0)).
- Parameters removed: None (retains alpha, gamma, w_features, w_familiar, and lapse).
- Other terms: choice_probs computes an utterance ambiguity cost as cost_ambiguity times the excess number of referents each non-sink word applies to beyond one, and passes it to L1; the simulated speaker utility in L1 subtracts this cost vector (component taken from costly_speaker_shared_prior); the decision rationality parameter gamma and the softmax belief decision rule in L1 and on uninformative prior trials are unchanged.
