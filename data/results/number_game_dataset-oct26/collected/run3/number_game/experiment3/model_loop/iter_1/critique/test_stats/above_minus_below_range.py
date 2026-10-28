# name: above_minus_below_range
# description: Among targets outside the set's [min,max] range by 1-30, yes-rate for targets above the max minus targets below the min; observed > null means people extend concepts upward (toward larger numbers) more than downward beyond what the model's interval prior predicts (observed < null: the reverse).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    a = (t > hi) & (t - hi <= 30)
    b = (t < lo) & (lo - t <= 30)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
