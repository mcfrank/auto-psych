We refine base_rate_simplicity_listener by incorporating the selective visual attention component from selective_attention_listener, positing that listeners selectively focus visual attention on candidate referents compatible with the heard utterance while discounting unattended distractors in the visual background. When simulating the speaker's communicative choices, the listener evaluates informativeness primarily over the attended candidates, so non-matching distractors exert attenuated influence on pragmatic inference across multi-referent scenes. In the absence of an informative utterance, the listener chooses directly according to the integrated familiarization base-rate and simplicity prior.

Differences from base_rate_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: distractor_attention (Beta(1.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes an attention mask assigning 1.0 to referents matching the heard utterance (ctx.lex[ctx.utterance] > 0) and distractor_attention to non-matching referents; passes this attention vector to L0 via L1; and in L0, the listener weights referent choices by at(lex, u, r) * vec(att, r).
