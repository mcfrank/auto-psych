When human listeners interpret referring expressions, capacity-limited visual attention across the visual scene is driven jointly by bottom-up visual salience and top-down empirical expectations learned from prior familiarization frequency. This experience-weighted attention directly gates literal semantic grounding, making frequently encountered referents more readily accessible to literal interpretation. Communicative speakers anticipate this attentional bias when selecting referring expressions, and pragmatic listeners invert this attention-grounded speaker at depth two to resolve referential ambiguity.

Refining limited_attention_listener by incorporating empirical familiarization base-rate sensitivity from rsa_l2_singleton_feat_color_val_fam_l0 into the capacity-limited visual attention distribution. This improves model fit by capturing participants' learned frequency expectations on familiarization trials (where prior exposure frequency directly shapes spontaneous referent expectations on uninformative trials and biases literal semantic grounding on informative trials) while maintaining depth-two recursive reasoning, attention-gated semantics, and perceptual salience.

Differences from source (limited_attention_listener):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: w_familiar ~ Normal(0.0, 2.0).
- Parameters removed: None.
- Terms changed: Exactly one term added: + params["w_familiar"] * ctx.familiarization inside the compute_attention softmax calculation. Literal L0 semantic gating, depth-2 recursive reasoning, and lapse process remain identical to the source.
