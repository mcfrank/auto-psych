# name: participant_bias_spread
# description: Standard deviation across the 15 participants (contiguous 450-trial blocks where trial_index restarts) of each participant's mean residual yes rate relative to the cell mean (same number_set and target, all participants); observed above null means stable individual differences in yes bias that the single pooled model cannot produce, below means people are more homogeneous than binomial noise.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, np.diff(ti) < 0])
    y = df["rating"].astype(float).to_numpy()
    cell = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    g = pd.Series(y).groupby(cell.to_numpy())
    s = g.transform("sum").to_numpy(); c = g.transform("count").to_numpy()
    ok = c > 1
    res = np.where(ok, y - (s - y) / np.maximum(c - 1, 1), np.nan)
    m = pd.Series(res).groupby(pid).mean()
    return float(np.nanstd(m.to_numpy()))
