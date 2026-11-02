# name: late_session_discrimination_change
# description: Correlation between responses and the leave-one-out cell mean of other participants (same number_set and target) in the last third of each session (trial_index >= 300) minus the first third (trial_index < 150); observed below null means responses become noisier / less stimulus-driven late in the session (fatigue or lapses the stationary model lacks), above means people become more consistent with practice.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    y = df["rating"].astype(float).to_numpy()
    cell = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    g = pd.Series(y).groupby(cell)
    s = g.transform("sum").to_numpy(); c = g.transform("count").to_numpy()
    ok = c > 1
    loo = (s - y) / np.maximum(c - 1, 1)
    early = ok & (ti < 150); late = ok & (ti >= 300)
    r1 = np.corrcoef(y[early], loo[early])[0, 1]
    r2 = np.corrcoef(y[late], loo[late])[0, 1]
    return float(r2 - r1)
