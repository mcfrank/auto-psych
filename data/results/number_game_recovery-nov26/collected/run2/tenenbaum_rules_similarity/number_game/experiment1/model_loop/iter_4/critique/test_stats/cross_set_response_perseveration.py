# name: cross_set_response_perseveration
# description: On trials whose set differs from the previous trial's set (same participant, participants split where trial_index resets), P(yes | previous answer yes) minus P(yes | previous answer no); the model is history-free, so observed above null means people carry over their last answer across set changes (response perseveration), below null means they alternate.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[1, (np.diff(ti) <= 0).astype(int)])
    s = df["number_set"].astype(str).to_numpy()
    r = df["rating"].to_numpy(dtype=float)
    same_p = np.r_[False, pid[1:] == pid[:-1]]
    diff_set = np.r_[False, s[1:] != s[:-1]]
    prev = np.r_[np.nan, r[:-1]]
    m = same_p & diff_set
    a = m & (prev == 1)
    b = m & (prev == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
