Listeners choose referents by applying a softmax decision rule over their posterior pragmatic beliefs rather than strict probability matching. When interpreting referring expressions in context, listeners reason about speakers who avoid competitor confusion and penalize unmentioned distinctive features, but sharpen their behavioral selections toward the most probable referent with a decision determinism parameter. On uninformative trials, choices default to the visual singleton salience prior, favoring contextually unique objects over duplicated items.

Refining rsa_l2_singleton_confusion_omission by incorporating a softmax decision rule (choice determinism) from softmax_belief_listener:
Listeners evaluate candidate referents via depth-2 pragmatic reasoning with visual singleton salience, competitor confusion avoidance, and omitted-alternative exhaustification, but convert their posterior beliefs into choices through a power-law softmax decision rule rather than probability matching. A decision determinism parameter gamma sharpens selections toward the most probable referent, capturing human choice determinism in forced-choice reference games.

Differences from rsa_l2_singleton_confusion_omission:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: gamma (LogNormal(0.0, 0.5), governing the decision determinism of the choice rule).
- Parameters removed: None.
- Other terms: In choice_probs, listener choice probabilities for informative queries are sharpened via a power-law softmax decision rule with exponent gamma (logits = gamma * log(heard + EPS); heard = softmax(logits)) before mixing with uniform guessing via lapse.
