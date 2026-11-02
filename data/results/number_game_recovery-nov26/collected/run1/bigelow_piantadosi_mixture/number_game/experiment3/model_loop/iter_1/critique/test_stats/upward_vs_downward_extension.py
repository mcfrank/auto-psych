# name: upward_vs_downward_extension
# description: Yes rate for targets 1-15 above the largest example minus yes rate for targets 1-15 below the smallest example (all set sizes); observed > null means people extend a range upward more than downward beyond what the model's symmetric interval prior predicts, observed < null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    E = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # (rows, 4), NaN padded
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    up = (t > hi) & (t <= hi + 15)
    dn = (t < lo) & (t >= lo - 15)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(y[up].mean() - y[dn].mean())
