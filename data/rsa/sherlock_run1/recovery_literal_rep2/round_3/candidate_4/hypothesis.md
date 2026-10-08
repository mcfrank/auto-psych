This model refines bounded_capacity_listener by proposing that listeners integrate visual color salience into their prior expectations over intended referents. When visual scenes contain objects differing in color presentation, full-color objects capture visual attention and are perceived as more salient than desaturated grayscale objects, leading listeners to discount grayscale alternatives. The pragmatic listener maintains bounded processing capacity, anchoring on the literal referent distribution and incorporating the speaker's communicative likelihood only to the extent permitted by their cognitive capacity.

Differences from bounded_capacity_listener:
- Refined model: bounded_capacity_listener
- Recursion depth: depth 1 (choice_probs calls L_bounded, unchanged).
- Parameters added: w_grayscale ~ Normal(0.0, 1.0) governing the perceptual penalty for desaturated grayscale objects in the prior over referents.
- Parameters removed: none.
- Prior calculation in choice_probs: adds params["w_grayscale"] * ctx.grayscale to the prior logits in softmax_prior.
- All other memo agents, distributions, and prior terms are unchanged.
