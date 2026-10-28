# name: previous_examples_carryover
# description: Yes-rate on non-example targets that were an example in the immediately preceding trial's set (same session, consecutive trial_index) minus the yes-rate on other non-example targets; observed above null means examples from the previous trial leak into the current concept (cross-trial contamination the trial-independent model cannot produce).
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    sets = df["number_set"].astype(str).str.replace(" ", "")
    s = ("," + sets + ",").to_numpy()
    t = df["target"].astype(int).astype(str).to_numpy()
    tok = np.array(["," + a + "," for a in t])
    prev_s = np.concatenate([[""], s[:-1]])
    consec = np.concatenate([[False], np.diff(ti) == 1])
    non_ex = np.array([a not in b for a, b in zip(tok, s)])
    in_prev = np.array([a in b for a, b in zip(tok, prev_s)]) & consec & (prev_s != s)
    r = df["rating"].to_numpy().astype(float)
    a = non_ex & in_prev
    b = non_ex & ~in_prev
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
