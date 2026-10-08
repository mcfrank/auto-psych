We refine costly_feature_speaker by introducing a feature-simplicity prior over referents, positing that speakers not only incur communicative costs when producing ambiguous words that apply to multiple referents, but also favor simpler referents with fewer distinguishing features. Listeners invert this ambiguity-averse, simplicity-sensitive speaker and rely directly on referent simplicity when no informative word is spoken.

Differences from costly_feature_speaker:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: simplicity (Normal(0.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes an object prior from feature counts via softmax_prior(simplicity * ctx.feature_count); L1 gives the speaker referents drawn from this simplicity prior (wpp=vec(prior, r) + {EPS} instead of wpp=1); and choice_probs uses this prior rather than a uniform distribution on uninformative prior trials (ctx.is_prior > 0).
