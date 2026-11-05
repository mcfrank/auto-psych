# name: parity_match_effect
# description: Among sets of >=2 examples that all share parity, yes-rate for targets of the same parity minus yes-rate for opposite-parity targets; observed above null_mean means people generalize by parity more strongly than the model, below means the model over-relies on odd/even.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, np.mod(ex, 2))
    allodd = np.nansum(par == 1, axis=1) == n
    alleven = np.nansum(par == 0, axis=1) == n
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    m = (n >= 2) & (allodd | alleven)
    setpar = np.where(allodd, 1, 0)
    match = (t % 2) == setpar
    a = m & match; b = m & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
