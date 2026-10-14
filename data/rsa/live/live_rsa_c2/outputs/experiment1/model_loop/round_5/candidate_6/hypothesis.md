We refine bounded_prevalence_redundancy_l2 by incorporating contextual end-point salience from endpoint_salience_l2 into the shared visual referent prior. In visual reference games, listeners structure their default expectations around contextual end-point salience: perceptual attention spontaneously anchors at the boundaries of the scene's feature complexity distribution (the minimal baseline object and the maximal exemplar), while intermediate objects that fall strictly between these poles are disfavored as unremarkable non-targets. This extremity prior directly rectifies the systematic overprediction of intermediate-feature objects on unprompted mumble trials and boosts maximal competitor exemplars during reference resolution.

Differences from bounded_prevalence_redundancy_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_endpoint (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Shared visual referent prior: Adds contextual endpoint salience (w_endpoint * is_endpoint) to the perceptual prior, where is_endpoint flags objects possessing either the minimum or maximum feature count in the display, disfavoring strictly intermediate objects.
- All other terms: Bounded capacity likelihood exponentiation (kappa) in L1 and L2, speaker contextual feature prevalence accessibility (w_prevalence) and contextual feature redundancy discounting (w_redundancy) on speaker alternatives in S1 and S2, literal truth-conditional semantics at L0, depth-2 RSA recursion, and uniform lapse mixing remain identical to bounded_prevalence_redundancy_l2.
