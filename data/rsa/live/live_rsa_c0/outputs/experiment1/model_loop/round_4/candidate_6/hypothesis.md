Listeners interpret referring expressions under capacity-limited visual attention where attentional engagement is modulated by referential ambiguity, balancing top-down pragmatic reasoning against bottom-up visual capture. Refining limited_attention_listener_2: while unambiguous expressions elicit focused top-down pragmatic verification, expressions that apply to multiple candidate referents dilute top-down attentional control, increasing the probability of bottom-up visual attentional capture by salient objects in the scene. This heightened capture under ambiguity increases selections of visually prominent non-matching foils and feature-rich objects on multi-match and ubiquitous displays.

Differences from limited_attention_listener_2:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L_attention, which inverts a speaker simulating depth-1 listener L1).
- Added parameter: w_ambiguity (prior dist.Normal(0.0, 1.0)).
- No parameters removed (retains alpha, w_confusion, logit_p_base, w_load, w_popout, w_singleton, w_distinct, w_features, and lapse with identical priors).
- Other terms: In choice_probs, computed referential ambiguity penalty term - w_ambiguity * ambiguity in logit_p_attend, where ambiguity is the number of competing matching referents beyond the target (matching_count - 1.0).
