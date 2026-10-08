We refine salience_structured_lapse_listener by proposing that listeners interpret referential expressions through parsimonious semantic comprehension, preferring candidate referents with fewer total features rather than treating all matching referents equally. When goal-directed attention lapses or when no informative clue is provided, choice is governed by a structured fallback to bottom-up visual and contextual salience, biasing clicks toward feature-rich and familiar objects. This parsimony preference naturally favors simpler referents in ambiguous communicative contexts while preserving salience-driven visual attention during lapses and prior trials.

Differences from source (salience_structured_lapse_listener):
- Recursion depth: Unchanged (choice_probs calls L0 at depth 0).
- Parameters added: beta (prior: LogNormal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L0, the listener's choice weight incorporates a parsimony discount against feature complexity, `wpp = at(lex, u, r) * vec(weight, r)`, where `weight = jnp.exp(-params["beta"] * ctx.feature_count)`, selecting among referents satisfying the heard word by preferring objects with fewer total features, while goal-directed lapses and prior trials continue to fall back to bottom-up contextual salience.
