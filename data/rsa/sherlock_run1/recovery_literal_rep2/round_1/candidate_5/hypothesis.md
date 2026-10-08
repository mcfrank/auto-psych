Depth-1 RSA with a shared salience prior and softmax decision rationality for the pragmatic listener: rather than strictly probability-matching posterior beliefs about the speaker's intended referent, listeners apply a softmax choice policy governed by an independent decision rationality parameter. This separates the speaker's communicative rationality from the listener's choice temperature, capturing the systematic conservatism observed in human object selection while preserving common knowledge of salience.

Differences from rsa_l1_shared_prior:
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: beta ~ LogNormal(0.0, 1.0) governing listener decision rationality.
- Parameters removed: none.
- Listener choice in L1: listener chooses r in OBJ with probability proportional to exp(beta * log(Pr[speaker.r == r] + EPS)) instead of matching Pr[speaker.r == r] directly.
- All other memo agents, distributions, and prior terms are unchanged.
