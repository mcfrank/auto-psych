Pragmatic listeners reason at depth 2 about speakers who avoid competitor confusion, penalize unmentioned informative features of the target, and account for visual singleton salience. When multiple descriptions apply to an object, speakers avoid using an ambiguous descriptor if the target possesses an unmentioned, highly discriminative alternative feature in the display, leading listeners to reject referents with superior unstated descriptors. On uninformative trials, listeners spontaneously favor unique visual singletons over duplicated items in the display.

Refining rsa_l2_singleton_confusion by incorporating omitted-alternative exhaustification from alternative_exhaustification_listener:
Speakers evaluate candidate referring expressions not only by informativeness to simulated listeners and distractor confusion avoidance, but also by penalizing expressions that leave more distinctive alternative features of the intended referent unsaid. This improves fit on displays where referents vary in contextual alternative availability, capturing human listeners' strong implicatures against targets with superior unmentioned descriptors while preserving visual singleton salience on uninformative displays.

Differences from rsa_l2_singleton_confusion:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_omission (Normal(0.0, 1.0), penalizing referring expressions that omit informative alternative features of the intended referent).
- Parameters removed: None.
- Other terms: Added compute_exhaustification_violation to quantify the contextual informativeness of unmentioned features for each candidate referent; incorporated - w_omission * at(violation, u, r) into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
