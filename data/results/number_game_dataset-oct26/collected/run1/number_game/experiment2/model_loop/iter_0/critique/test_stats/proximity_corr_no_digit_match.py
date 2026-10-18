# name: proximity_corr_no_digit_match
# description: Correlation between rating and -log(1+distance to nearest example) among targets sharing no final digit with any example; observed above model mean means numeric closeness drives yes beyond what the digit model predicts.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    nodig = ~((X % 10) == (t[:, None] % 10)).any(axis=1) & (dist > 0)
    x = -np.log1p(dist[nodig]); y = r[nodig]
    if len(y) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
