NOTE: this is a repair attempt. Your previous attempt at this candidate slot, in `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment3/model_loop/iter_4/candidate_5`, was rejected at admission for this reason:

    MCMC sampling failed (RuntimeError: warm-start NUTS: the Hessian at the posterior mode is not positive definite (some parameter direction is pinned down by neither the data nor the prior)); dropping it so it cannot abort scoring.

Its `hypothesis.md`, `model_name.txt` and `candidate.py` have been copied into your working directory as a starting point. Fix what the reason describes — a different mechanism if the model predicts like an existing one, corrected code if it does not load, fit or score, a cheaper model if it was too slow to fit — and write all three files again to the paths above. A second rejection is final: there is no further attempt at this slot.
