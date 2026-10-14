Pragmatic listeners reason at depth 1 with feature-present singleton salience, attributing perceptual pop-out only to unique objects that possess actual visual features rather than to featureless, blank objects. Refining rsa_l1_singleton_feat_color_familiar, visual singleton salience is restricted to items with non-zero feature counts, reflecting the psychological asymmetry that visual search detects the presence of distinguishing features rather than the absence of features. This directly corrects the incumbent model's severe overprediction of empty singleton choices and singleton rates in displays containing duplicated competitors, while maintaining first-order pragmatic reasoning over contextual distinctiveness, visual complexity, color contrast, and familiarization base rates.

Differences from source (rsa_l1_singleton_feat_color_familiar):
- Recursion depth: depth 1 (calls L1, unchanged from rsa_l1_singleton_feat_color_familiar).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In the referent prior, discrete visual singleton salience is gated by the presence of visual features (multiplying is_singleton by an indicator that feature count exceeds zero), ensuring empty objects without duplicates do not receive singleton pop-out bonuses.
