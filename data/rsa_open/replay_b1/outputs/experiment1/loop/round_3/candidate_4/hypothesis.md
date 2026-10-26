Speakers choose referring expressions to maximize contextual contrast against distinct competitor types in the immediate display, weighting each distractor's contribution inversely by its duplicate multiplicity rather than accumulating contrast independently across identical clones. Listeners interpret utterances by inverting this type-contrastive speaker, expecting heard words to discriminate the target from each competing distractor category without artificially penalizing shared features in displays with background duplicates. On uninformative displays without a referring expression, visual singleton salience guides referent expectations.

Refining contextual_contrast_listener with type-weighted contrast directly addresses the discrepancy in multiclone competitor displays (multiclone_competitor_margin_unit20), where summing contrast over duplicate tokens over-inflates the speaker's contrast utility for private features and wrongly predicts that a singleton competitor will receive fewer points than identical clones.

## Differences from contextual_contrast_listener

1. **Recursion depth**: Unchanged (depth 1, listener calls `L_contrast`).
2. **Parameters**: Unchanged (`PARAMS` contains `alpha` and `w_singleton`; no parameters added or removed).
3. **Contrast functional form**: In `listener(params, ctx)`, distractor similarity `sim` is normalized by distractor duplicate multiplicity (`type_sim = sim / duplicate_count[:, None]`), and contextual contrast is evaluated against distractor types (`contrast = real_lex * ((1.0 - real_lex) @ type_sim)`) rather than unnormalized tokens (`contrast = real_lex * ((1.0 - real_lex) @ sim)`). All prior terms, likelihood definitions, and memo structures are unchanged.
