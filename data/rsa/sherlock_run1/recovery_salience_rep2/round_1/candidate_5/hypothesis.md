Listeners reason at depth 2 while integrating familiarization base rates into their prior expectations about which object a speaker intends to refer to. Rather than assuming all objects are equally likely a priori, listeners blend base-rate exposure with a uniform baseline, weighting referents accordingly when interpreting the depth-2 speaker and when guessing on uninformative trials.

Differences from rsa_l2:
- Recursion depth: depth 2 (calls L2, unchanged from rsa_l2).
- Parameters added: base_rate_weight (Beta(2.0, 2.0), prior weight on familiarization base rates); alpha and lapse are unchanged.
- Mechanism: in L2, the listener inverts the speaker using a referent prior shaped by familiarization base rates instead of an unconditional uniform prior; choice_probs returns this prior on uninformative trials.
