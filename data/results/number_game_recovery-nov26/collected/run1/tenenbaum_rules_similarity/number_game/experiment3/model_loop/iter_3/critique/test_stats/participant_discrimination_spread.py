# name: participant_discrimination_spread
# description: Standard deviation across participants (450-trial blocks) of the correlation between a participant's responses and the leave-one-out mean response of the other participants on the same (number_set, target) cell; observed above null means participants differ in how closely they follow the shared generalisation pattern (individual differences in strategy or attentiveness the pooled model lacks), below means more uniform than predicted.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = np.cumsum(np.r_[0, np.diff(ti) < 0])
    y = df["rating"].astype(float).to_numpy()
    cell = (df["number_set"].astype(str) + "|" + df["target"].astype(str)).to_numpy()
    g = pd.Series(y).groupby(cell)
    s = g.transform("sum").to_numpy(); c = g.transform("count").to_numpy()
    ok = c > 1
    loo = (s - y) / np.maximum(c - 1, 1)
    t = pd.DataFrame({"p": pid[ok], "y": y[ok], "l": loo[ok]})
    t["yl"] = t.y * t.l; t["yy"] = t.y ** 2; t["ll"] = t.l ** 2
    a = t.groupby("p").mean()
    cov = a.yl - a.y * a.l
    sd = np.sqrt(np.maximum(a.yy - a.y ** 2, 1e-12) * np.maximum(a.ll - a.l ** 2, 1e-12))
    r = (cov / sd).to_numpy()
    return float(np.std(r))
