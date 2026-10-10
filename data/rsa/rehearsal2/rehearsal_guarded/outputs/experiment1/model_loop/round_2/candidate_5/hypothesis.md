Listeners interpret referring expressions by inverting a speaker who both avoids expressions that create competitor confusion and penalizes referring expressions that omit informative alternative features of the target referent. When multiple descriptions apply to an object, speakers avoid using a less informative word if the object possesses an unmentioned, highly discriminative alternative feature in the context, leading listeners to reject referents with superior unstated descriptors. This omitted-alternative exhaustification penalty operates alongside competitor confusion avoidance and visual salience at depth-2 pragmatic recursion.

Refining rsa_l2_salience_confusion_l0 by incorporating omitted-alternative exhaustification from alternative_exhaustification_listener:
Speakers evaluate candidate referring expressions not only by informativeness to simulated listeners and visual confusion with distractors, but also by penalizing expressions that leave more distinctive alternative features of the target unsaid. When a target object possesses a highly unique, unmentioned feature, choosing a shared or ambiguous descriptor incurs a communicative omission penalty (violating Gricean quantity), capturing human listeners' strong implicatures on displays where referents differ in contextual alternative availability.

Differences from rsa_l2_salience_confusion_l0:
- Recursion depth: unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_omission (Normal(0.0, 1.0)), weighting the penalty against referring expressions that omit informative alternative features of the intended referent.
- Parameters removed: none.
- Other terms: added compute_exhaustification_violation to quantify the contextual informativeness of unmentioned features for each candidate referent; incorporated - w_omission * at(violation, u, r) into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
