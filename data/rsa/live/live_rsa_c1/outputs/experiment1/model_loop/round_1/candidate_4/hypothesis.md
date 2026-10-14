RSA pragmatic listener at depth 2 with salience prior and softmax belief choice rule.

Refining rsa_l2_singleton_feat_color_familiar_l0 with the softmax belief choice rule component from softmax_surprisal_isolated_costly_l4: listeners reason at depth 2 about a speaker who simulates a depth-1 pragmatic listener with a shared referent prior over singleton salience, contextual distinctiveness, feature complexity, color contrast, and familiarization frequency, but translate their posterior pragmatic beliefs into choice probabilities via a power-law softmax response rule rather than probability matching. This bounded-rationality response rule softens over-confident pragmatic inferences, directly resolving the model's empirical overprediction of minimal-feature targets in ambiguous utterance trials with duplicated competitors or large feature gaps. On uninformative prior trials, choices directly follow the perceptual and experiential prior, and uniform guessing is mixed in via a lapse parameter.

Differences from source (rsa_l2_singleton_feat_color_familiar_l0):
- Recursion depth: Unchanged at depth 2; choice_probs evaluates L2.
- Parameters added: beta (prior: dist.LogNormal(0.0, 1.0)), governing the softmax belief choice rule.
- Parameters removed: None.
- Choice computation: In choice_probs on utterance trials, the listener posterior distribution heard is transformed into choice by exponentiating log beliefs scaled by beta and renormalizing, rather than using heard directly.
