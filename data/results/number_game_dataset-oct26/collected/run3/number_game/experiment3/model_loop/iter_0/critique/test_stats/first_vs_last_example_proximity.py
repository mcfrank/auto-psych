# name: first_vs_last_example_proximity
# description: In sets of 2+ examples, yes-rate for targets within 5 of the first-shown example (but more than 5 from the last-shown) minus targets within 5 of the last-shown (but more than 5 from the first), excluding set members; observed > 0 relative to null means a primacy effect (observed < null: recency) that the order-blind model cannot produce.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    idx = np.clip(n - 1, 0, None).astype(int)
    first = E[:, 0]
    last = E[np.arange(len(E)), idx]
    inset = np.any(E == t[:, None], axis=1)
    m = (n >= 2) & ~inset
    nf = np.abs(t - first) <= 5; nl = np.abs(t - last) <= 5
    a = m & nf & ~nl; b = m & nl & ~nf
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
