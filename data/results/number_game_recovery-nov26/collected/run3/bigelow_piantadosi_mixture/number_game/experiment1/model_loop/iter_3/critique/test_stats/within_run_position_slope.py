# name: within_run_position_slope
# description: OLS slope of yes (non-example targets) on log(1 + position within a run of consecutive trials showing the same set); observed > null means people grow more liberal as they keep answering about one set (observed < null more conservative) than the static model produces.
def test_statistic(df):
    seq = df[["trial_index", "number_set"]].drop_duplicates("trial_index").sort_values("trial_index")
    newrun = (seq["number_set"] != seq["number_set"].shift()).values
    run_id = np.cumsum(newrun)
    pos = pd.Series(seq["trial_index"].values).groupby(run_id).cumcount().values
    pmap = pd.Series(pos, index=seq["trial_index"].values)
    p = np.log1p(df["trial_index"].map(pmap).values.astype(float))
    padded = ("," + df["number_set"].astype(str) + ",").values
    tstr = ("," + df["target"].astype(int).astype(str) + ",").values
    member = np.array([b.find(a) >= 0 for a, b in zip(tstr, padded)])
    x, y = p[~member], df["rating"].astype(float).values[~member]
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
