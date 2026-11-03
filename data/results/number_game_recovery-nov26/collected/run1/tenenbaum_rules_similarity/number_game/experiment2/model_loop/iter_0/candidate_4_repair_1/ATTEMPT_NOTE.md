NOTE: this is a repair attempt. Your previous attempt at this candidate slot, in `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/iter_0/candidate_4`, was rejected at admission for this reason:

    predicts like existing model 'erlang_interval_size_family_prior' (p_yes RMSE 0.00053 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of erlang_interval_size_family_prior, not a new hypothesis.

Its `hypothesis.md`, `model_name.txt` and `candidate.py` have been copied into your working directory as a starting point. Fix what the reason describes — a different mechanism if the model predicts like an existing one, corrected code if it does not load, fit or score, a cheaper model if it was too slow to fit — and write all three files again to the paths above. A second rejection is final: there is no further attempt at this slot.
