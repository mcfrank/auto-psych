Literal semantic grounding is directly gated by visual salience and learned base-rate expectations, so that words preferentially bring salient referents to mind even before pragmatic reasoning begins. Speakers anticipate this attention-grounded interpretation and avoid referring expressions shared with confusable competitors, while pragmatic listeners recursively invert this speaker to resolve referring expressions.

Refining rsa_l2_competitor_confusion_fam_l0 by incorporating the attention-grounded literal semantics mechanism from limited_attention_listener into the foundational L0 listener. This should fit better by aligning literal semantic applicability with contextual salience and learned familiarization frequency across all displays, explaining participants' strong tendency to resolve ambiguous words toward visually salient and frequent referents while preserving depth-two recursive reasoning and competitor-confusion aversion.

Differences from source (rsa_l2_competitor_confusion_fam_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: None.
- Parameters removed: None.
- Terms changed: L0 literal listener choice weights become wpp=at(lex, u, r) * vec(prior, r) (incorporating the shared perceptual and empirical prior into literal semantic grounding, taken from limited_attention_listener), with prior passed as an argument to L0 in L1. The prior computation, competitor confusion matrix calculation, S1 and S2 speaker choice utilities, depth-2 recursive reasoning, and lapse process remain identical to the source.
