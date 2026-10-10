Listeners interpret referring expressions using a twin-contrast perceptual oddity heuristic, where an item receives an oddity pop-out boost only when it constitutes a solitary singleton contrasting specifically against an identical twin pair in the visual scene. When identical background items form a larger triplet majority, the category base rate balances visual contrast and listeners choose uniformly among matching items.

Refining solitary_oddity_heuristic_listener: The source model granted an oddity pop-out boost to any singleton that was the sole unique item in the context, regardless of whether the background duplicates formed a pair or a larger triplet majority. We restrict the oddity boost to singletons contrasting specifically against an identical twin pair (copy count of 2), so that triplet-majority displays default to uniform choice among matching items.

Differences from source:
- Recursion depth: Unchanged at depth 0 (choice_probs evaluates L_heur directly without recursive speaker simulation).
- Parameters added or removed: None (retains parameter beta with prior Normal(0.0, 2.0) and parameter lapse with prior Beta(1.0, 9.0)).
- Code differences: In compute_singleton_indicator, max_copy is computed via jnp.max(copy_count), and is_solitary_singleton requires (max_copy > 1.5) & (max_copy < 2.5) in addition to is_item_singleton and singleton_count.
