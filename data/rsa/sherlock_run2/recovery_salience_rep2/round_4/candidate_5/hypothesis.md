Pragmatic listeners at depth 2 expect speakers to incur an information-theoretic ambiguity cost that exhibits diminishing sensitivity to the number of competing referents an utterance describes, reflecting the logarithmic entropy of ambiguous candidate sets. Rather than penalizing each additional competitor equally, speakers experience the greatest communicative penalty when shifting from a unique to a shared label, with diminishing marginal penalties for subsequent competitors. This diminishing ambiguity cost guides second-order recursive speaker simulation alongside inductive base-rate and cognitive parsimony priors, explaining why listeners penalize multi-referent descriptions without over-penalizing expressions on crowded displays.

### Refinement of salience_base_rate_ambiguity_l2
We refine salience_base_rate_ambiguity_l2 with a single stated change: replacing the linear speaker ambiguity cost with a logarithmic ambiguity cost reflecting the referential entropy of candidate extensions, capturing diminishing marginal cognitive penalties as the competitor set grows.

### Differences from salience_base_rate_ambiguity_l2
- Recursion depth: depth 2 (choice_probs calls L2, unchanged from salience_base_rate_ambiguity_l2).
- Parameters added: none.
- Parameters removed: none (alpha, cost, base_rate_weight, salience, and lapse retained with identical priors).
- Utterance costs: choice_probs computes costs using a logarithmic functional form based on referential entropy (costs = params["cost"] * jnp.log(jnp.maximum(1.0, extension))) instead of a linear excess-count penalty (costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)), leaving cost at zero for unambiguous utterances and sink tokens while scaling logarithmically with competitor set size.
- Memo recursion: unchanged memo signatures and agent definitions; L0, L1, and L2 retain identical definitions, and choice_probs continues to pass the inductive prior and costs into L2.
