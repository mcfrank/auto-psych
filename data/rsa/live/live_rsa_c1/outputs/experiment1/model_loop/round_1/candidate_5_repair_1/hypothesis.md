Pragmatic listeners interpret utterances under graded semantic applicability, reasoning that words are not rigidly binary truth-conditional filters but have feature-dependent applicability that scales with a referent's total visual feature complexity. By capturing graded semantic typicality at the foundational literal listener level, listeners avoid over-penalizing competitor referents that possess multiple additional features or duplicate copies when hearing a shared descriptor.

Refining rsa_l2_singleton_feat_color_familiar_l0 by incorporating the graded semantics component from softmax_surprisal_isolated_costly_l4:
- Recursion depth: Depth 2 (choice_probs evaluates L2, unchanged from rsa_l2_singleton_feat_color_familiar_l0).
- Parameters added: gamma (prior: dist.Normal(0.0, 0.5)), governing feature-dependent semantic applicability.
- Parameters removed: None.
- Other terms: A graded lexicon is computed via compute_graded_lex from ctx.lex and ctx.feature_count and passed to L0, L1, and L2, replacing binary truth values while preserving singleton salience, contextual distinctiveness, feature complexity, color contrast, and familiarization base rates in the referent prior.
