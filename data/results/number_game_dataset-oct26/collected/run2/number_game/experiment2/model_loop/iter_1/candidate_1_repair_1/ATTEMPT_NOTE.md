NOTE: this is a repair attempt. Your previous attempt at this candidate slot, in `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/iter_1/candidate_1`, was rejected at admission for this reason:

    model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

Its `hypothesis.md`, `model_name.txt` and `candidate.py` have been copied into your working directory as a starting point. Fix what the reason describes — a different mechanism if the model predicts like an existing one, corrected code if it does not load, fit or score, a cheaper model if it was too slow to fit — and write all three files again to the paths above. A second rejection is final: there is no further attempt at this slot.
