Pragmatic listeners reason at depth two about a speaker who avoids referring expressions shared with visually confusable competitors, while sharing an object prior that integrates perceptual salience with empirical base rates learned from familiarization frequency. Prior exposure establishes expectations that directly bias referent selection on both uninformative and informative trials, combining additively with visual singleton salience, contextual distinctiveness, evaluative valence-modulated feature complexity, and color contrast.

Refining rsa_l2_competitor_confusion_l0 by incorporating empirical familiarization base-rate sensitivity from rsa_l2_singleton_feat_color_val_fam_l0 into the shared object prior. This should fit better by capturing participants' learned frequency expectations on familiarization trials (such as the base-rate conditions where uninformative and informative choices strongly track prior exposure frequency) while preserving depth-two recursive reasoning, competitor-confusion aversion, and perceptual salience.

Differences from source (rsa_l2_competitor_confusion_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: w_familiar ~ Normal(0.0, 2.0).
- Parameters removed: None.
- Terms changed: Exactly one term added: + params["w_familiar"] * ctx.familiarization inside the softmax prior computation. The competitor confusion matrix calculation, S1 and S2 speaker choice utilities, literal L0 semantics, depth-2 recursive reasoning, and lapse process remain identical to the source.
