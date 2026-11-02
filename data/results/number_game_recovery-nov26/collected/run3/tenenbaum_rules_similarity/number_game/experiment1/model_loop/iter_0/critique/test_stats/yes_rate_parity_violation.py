# name: yes_rate_parity_violation
# description: Among sets whose examples all share one parity, the yes rate for targets of the opposite parity; observed above the model's mean means people generalize across parity more than the odd/even rules allow.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        E[i, :len(v)] = v
    X = E[codes]  # (rows, 4) examples, NaN padded
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(X), axis=1)
    par = np.where(np.isnan(X), np.nan, X % 2)
    lo = np.nanmin(par, axis=1); hi = np.nanmax(par, axis=1)
    m = (lo == hi) & ((t % 2) != lo)
    return float(r[m].mean()) if m.any() else 0.0
