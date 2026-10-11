Pragmatic listener at depth 2 with capacity-limited attention, expected communicative success, competitor confusion avoidance, and utterance specificity costs. Refining expected_success_confusion_attention by incorporating the utterance specificity cost component from specificity_cost_speaker: speakers weigh communicative success against a cognitive specificity cost for producing highly distinguishing referring expressions, frequently choosing broader or ubiquitous words even when describing feature-rich objects. Under capacity-limited visual attention, pragmatic listeners invert this cost-sensitive, confusion-averse speaker, moderating overly aggressive scalar exclusion and elevating choices of feature-rich referents on ubiquitous utterance and two-object scalar displays.

Differences from expected_success_confusion_attention:
- Recursion depth remains depth 2 (choice_probs calls L_attention, which inverts a speaker simulating depth-1 listener L1).
- Added parameter: c_spec (prior dist.Normal(0.0, 1.0)).
- No parameters removed (retains alpha, logit_p_base, w_load, w_popout, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse).
- Speaker utility in both S1 and S2: added utterance specificity cost term - vec(cost, u) inside communicative utility, evaluating as alpha * (L - vec(cost, u)) - w_confusion * at(confusion, u, r).
- In choice_probs, computed utterance extension and specificity cost vector cost = params["c_spec"] * (1.0 - extension) * (1.0 - ctx.is_sink), passing cost to L1 and L_attention.
