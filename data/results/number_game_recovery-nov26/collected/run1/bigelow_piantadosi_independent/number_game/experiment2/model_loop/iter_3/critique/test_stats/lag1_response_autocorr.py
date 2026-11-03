# name: lag1_response_autocorr
# description: Correlation between consecutive responses within a session (sessions split where trial_index resets, rows ordered by trial_index); observed > null means responses are serially perseverative beyond what the trial-independent model produces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, (np.diff(ti) < 0).astype(int)])
    order = np.lexsort((ti, pid))
    y = df["rating"].astype(float).to_numpy()[order]
    p = pid[order]
    same = p[1:] == p[:-1]
    a, b = y[:-1][same], y[1:][same]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
