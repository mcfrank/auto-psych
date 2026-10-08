Listeners incorporate empirical familiarization base rates into an inductive prior over referents rather than assuming objects are equally likely a priori. This prior guides referent expectations when reasoning recursively about speakers at depth 2, and serves as the listener's default choice distribution when no informative word is uttered. Compared to rsa_l2, which enforces a uniform prior across all displays, weighting referents by prior exposure frequency explains systematic preferences on both base-rate trials and uninformative prior trials.

Differences from rsa_l2:
- Recursion depth: depth 2 (choice_probs calls L2, unchanged from rsa_l2).
- Parameter added: base_rate_weight (prior dist.Normal(0.0, 2.0)), which scales sensitivity to familiarization base rates; no parameters removed (alpha and lapse retained with identical priors).
- Prior in memo recursion: added array parameter prior: ... to L1 and L2, replacing speaker: given(r in OBJ, wpp=1) with speaker: given(r in OBJ, wpp=vec(prior, r)) in both levels, and forwarding prior from L2 to L1.
- Choice probabilities: choice_probs computes prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization), passes prior to L2, and uses prior instead of uniform on prior trials (jnp.where(ctx.is_prior > 0, prior, heard)).
