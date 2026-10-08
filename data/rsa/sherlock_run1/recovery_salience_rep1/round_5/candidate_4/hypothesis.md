We refine prior_lapse_focal_listener by having the literal listener directly integrate empirical familiarization base rates into literal referent evaluation alongside focal attentional discounting. Rather than assuming the speaker simulates a literal listener who treats candidate objects with uniform environmental expectations, the speaker anticipates a Bayesian literal listener who already favors environmentally frequent referents. This sharpens communicative incentives by requiring speakers to use distinctive labels for rare referents while allowing concise, shared descriptions for familiar ones, while preserving prior-anchored lapse fallback and focal distractor discounting.

Differences from prior_lapse_focal_listener:
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from prior_lapse_focal_listener).
- Parameters added: None (alpha, salience, baserate, distractor_weight, and lapse retained unchanged).
- Parameters removed: None.
- Functional terms changed:
  - In L0, added array parameter base_prior: ... and weighted literal listener referent selection by at(lex, u, r) * (vec(base_prior, r) + {EPS}) * (vec(atten, r) + {EPS}), incorporating familiarization base rates into literal interpretation.
  - In L1, passed base_prior: ... through to L0.
  - In choice_probs, computed base_prior = softmax_prior(params["baserate"] * fam_logits) and passed base_prior into L1.
