The population of listeners is cognitively heterogeneous, consisting of a mixture of literal listeners and depth-2 pragmatic listeners who invert an informative speaker that simulates pragmatic counterfactual inferences. While depth-2 pragmatic listeners softly maximize over the communicative beliefs of a speaker anticipating pragmatic interpretations, literal listeners simply choose among candidate referents satisfying the literal semantics of the utterance. This fits better because higher-order recursive mentalizing resolves scalar implicatures and competitor distinctions that depth-1 reasoning under-differentiates, while preserving the population mixture over reasoning types.

Refinement of literal_pragmatic_mixture_listener:
- Base model refined: literal_pragmatic_mixture_listener
- Recursion depth: Depth 2 (choice_probs calls L0 and L2, replacing L1).
- Parameters added: None (retains alpha, gamma, p_pragmatic, w_features, w_familiar, and lapse).
- Parameters removed: None.
- Other terms: choice_probs calls L2 instead of L1 for the pragmatic subpopulation, adding the L2 listener memo that inverts an informative speaker S2 who simulates depth-1 pragmatic listener L1; the literal listener L0, population mixture proportion p_pragmatic, softmax belief precision gamma in L2, and prior-trial choices are unchanged.
