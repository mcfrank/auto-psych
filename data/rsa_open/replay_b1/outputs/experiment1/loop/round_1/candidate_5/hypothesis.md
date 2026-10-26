# rsa_l2_uncompounded_prior_l0

Pragmatic listener at depth 2 with uncompounded perceptual prior and literal L0. Listeners invert a pragmatic speaker who simulates a purely semantic, uncompounded pragmatic listener, integrating the visual perceptual prior solely at the top decision level rather than redundantly compounding perceptual biases across recursive reasoning tiers. On uninformative prior trials without an informative word, listener choices follow the perceptual prior directly.

## Refinement of rsa_l2_singleton_feat_color_valence_l0

This model refines `rsa_l2_singleton_feat_color_valence_l0` with ONE stated change: removing the perceptual prior from the intermediate simulated pragmatic listener L1 so that L1 reasons with a uniform object prior. In the source model, L1 used the perceptual prior, which was then exponentiated in S2's communicative utility and multiplied again by the perceptual prior at L2, redundantly compounding perceptual biases across tiers. This compounding swamped semantic informativeness, leading to severe discrepancies where listeners were incorrectly predicted to prefer singleton competitors over duplicate scalar targets (e.g., in `singleton_competitor_inversion_unit25` and `universal_feature_simplicity_advantage_unit35`). Uncompounding the prior ensures that the visual prior (singleton salience, distinctiveness, complexity, and color contrast) is integrated exactly once at the top-level listener, preserving pragmatic scalar implicatures while maintaining perceptual salience.

## Differences from source

1. **Recursion depth**: Unchanged (both call depth-2 listener `L2`).
2. **Parameters**: None added or removed (`PARAMS` is identical: `alpha`, `w_singleton`, `w_distinct`, `w_features`, `w_valence`, `w_color`).
3. **Definition of L1**: In `L1`, the object prior `speaker: given(r in OBJ, wpp=vec(prior, r))` is replaced with a uniform prior `speaker: given(r in OBJ, uniformly)`, and the `prior` parameter is removed from `L1`'s signature (`L1[u: UTT, r: OBJ](alpha, lex: ...)`).
4. **Call to L1 in L2**: The call inside `L2`'s speaker utility is updated from `L1[u, r](alpha, lex, prior)` to `L1[u, r](alpha, lex)`.
5. **Other terms**: Unchanged. `L0`, `L2`, the perceptual prior feature computation in `listener`, and the prior trial handling `jnp.where(ctx.is_prior > 0, prior, heard)` are identical.
