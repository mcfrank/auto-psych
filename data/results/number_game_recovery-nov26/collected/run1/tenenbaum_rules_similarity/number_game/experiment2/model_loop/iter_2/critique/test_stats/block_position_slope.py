# name: block_position_slope
# description: OLS slope of yes on the within-block position (0 = first target shown for that set in that run of trials, scaled to 0-1 within the block), across blocks of >=5 trials; observed above/below null means people grow more liberal/more conservative as they see more targets for the same set, beyond what the target identities imply.
def test_statistic(df):
    s = df["number_set"].to_numpy()
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    new = np.ones(len(df), dtype=bool)
    new[1:] = (s[1:] != s[:-1]) | (ti[1:] != ti[:-1] + 1)
    bid = np.cumsum(new)
    g = pd.Series(np.arange(len(df))).groupby(bid)
    pos = g.cumcount().to_numpy().astype(float)
    size = g.transform("size").to_numpy().astype(float)
    m = size >= 5
    x = pos[m] / (size[m] - 1)
    y = r[m]
    if m.sum() < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
