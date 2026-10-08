People interpret referring expressions by evaluating statistical preemption beliefs through a softmax decision rule rather than linear probability matching. When hearing an unspecific referring expression, listeners evaluate direct cue competition among alternative candidate words, penalizing referents whose distinctive features were bypassed, and softly maximize over these resulting posterior beliefs according to a decision rationality parameter. This mechanism sharpens referent selection toward the least preempted candidate while naturally accounting for bounded decision precision.

Refinement of statistical_preemption_listener:
- Base model refined: statistical_preemption_listener
- Recursion depth: Depth 0 / direct preemption listener (choice_probs calls L_preempt, unchanged from statistical_preemption_listener).
- Parameters added: gamma (prior: LogNormal(0.0, 1.0)).
- Parameters removed: None (retains beta, w_contrast, w_familiar, w_color, and lapse).
- Other terms: The listener evaluates preemption-discounted candidate referents using a softmax decision rule with decision rationality gamma, choosing with probability proportional to exp(gamma * L_preempt_base) rather than linear probability matching, and uninformative prior choices softly maximize over visual salience via softmax(gamma * prior) rather than prior (component taken from softmax_belief_listener); the direct word specificity calculation, competitor preemption metric, and diagnostic feature contrast prior are unchanged.
