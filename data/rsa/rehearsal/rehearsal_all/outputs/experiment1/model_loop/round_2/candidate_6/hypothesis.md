Pragmatic listeners reason at depth two about a speaker who actively avoids referring expressions that deprive competitor referents of their sole descriptive labels. People expect communicative speakers to choose alternative descriptive words when describing multi-feature items in order to preserve shared labels that feature-sparse competitors exclusively rely on. Pragmatic listeners invert this competitor-reliance-sensitive speaker while integrating common knowledge of an object prior combining discrete singleton salience, contextual distinctiveness, valence-modulated feature complexity, color contrast, and familiarization base rates.

Refining rsa_l2_singleton_feat_color_val_fam_l0 by incorporating competitor-reliance sensitivity from competitor_reliance_speaker into the speaker choice utilities. This improves model fit by accounting for participants' sensitivity to alternative descriptive availability across competitors in complex multi-object displays while preserving depth-two recursive reasoning, truth-conditional foundational semantics, and empirical familiarization expectations.

Differences from source (rsa_l2_singleton_feat_color_val_fam_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: w_reliance ~ Normal(0.0, 2.0).
- Parameters removed: None.
- Terms changed: S1 and S2 speaker choice utilities subtract w_reliance * at(reliance, u, r), where reliance is the competitor-reliance matrix computed from the non-sink lexicon normalized by competitor feature counts. The prior computation, literal L0 semantics, and lapse process remain identical to the source.
