# name: cell_unanimity_rate
# description: Fraction of (set, target) cells with >=5 responses in which all responses agree (all yes or all no); observed above null means people's per-item yes probabilities are more extreme than the model's (model too noisy/hedged), below null means people are less consistent across participants than the model.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    key = df["number_set"].astype(str).values + "|" + df["target"].astype(str).values
    codes, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    s = np.bincount(inv, weights=r)
    ok = cnt >= 5
    if not ok.any():
        return 0.0
    un = (s[ok] == 0) | (s[ok] == cnt[ok])
    return float(un.mean())
