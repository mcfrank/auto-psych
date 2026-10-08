We refine rsa_l1_salience by proposing that affective framing modulates object salience: listeners expect a speaker describing their favorite object to favor feature-rich items, whereas a speaker describing their least favorite object attenuates or reverses this feature-rich preference.

Differences from source rsa_l1_salience:
- Model refined: rsa_l1_salience
- Recursion depth: unchanged (choice_probs calls L1)
- Parameters added: w_valence with prior dist.Normal(0.0, 1.0)
- Parameters removed: none
- Other terms: in choice_probs, the object prior logits include an affective framing term modulating feature salience by speaker valence, w_valence * ctx.valence * ctx.feature_count.
