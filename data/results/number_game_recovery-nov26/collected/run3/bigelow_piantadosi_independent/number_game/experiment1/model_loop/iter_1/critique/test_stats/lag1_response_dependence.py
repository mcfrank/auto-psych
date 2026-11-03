# name: lag1_response_dependence
# description: P(yes | previous trial in the same session was yes) minus P(yes | previous was no), with sessions delimited by trial_index resets; observed > null means responses are sequentially perseverative, which the trial-independent model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    same = np.r_[False, ti[1:] > ti[:-1]]
    prev = np.r_[np.nan, r[:-1]]
    a = same & (prev == 1); b = same & (prev == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
