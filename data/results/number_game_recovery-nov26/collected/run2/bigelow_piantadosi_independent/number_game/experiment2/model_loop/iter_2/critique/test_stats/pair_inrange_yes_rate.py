# name: pair_inrange_yes_rate
# description: For 2-example sets, the yes rate for targets lying between the two examples (inclusive); observed below null means people treat the interval spanned by a pair more conservatively than the model's capacity-limited interval posterior predicts, above means they fill in the pair's range more readily.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    m = (n == 2) & (t >= lo) & (t <= hi)
    return float(y[m].mean()) if m.any() else 0.0
