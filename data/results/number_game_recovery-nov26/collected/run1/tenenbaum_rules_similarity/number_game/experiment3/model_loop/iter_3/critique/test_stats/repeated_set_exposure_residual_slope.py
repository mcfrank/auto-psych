# name: repeated_set_exposure_residual_slope
# description: Correlation between how many times a participant has previously seen the same number_set in their session and the response residual (rating minus the leave-one-out cell mean of other participants), among trials of sets the participant sees at least 5 times; observed below null means people narrow their generalisation (more no) as a set is re-encountered, above means they broaden — set-specific learning the stationary model lacks.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, np.diff(ti) < 0])
    y = df["rating"].astype(float).to_numpy()
    ns = df["number_set"].astype(str).to_numpy()
    cell = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    g = pd.Series(y).groupby(cell)
    s = g.transform("sum").to_numpy(); c = g.transform("count").to_numpy()
    res = y - (s - y) / np.maximum(c - 1, 1)
    key = pd.Series(pid.astype(str)) + "#" + pd.Series(ns)
    k = pd.DataFrame({"key": key, "ti": ti}).sort_values(["key", "ti"])
    exp_ = k.groupby("key").cumcount().reindex(range(len(df))).to_numpy()
    tot = key.map(key.value_counts()).to_numpy()
    ok = (c > 1) & (tot >= 5)
    return float(np.corrcoef(np.log1p(exp_[ok]), res[ok])[0, 1])
