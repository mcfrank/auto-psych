# name: dist2_minus_dist1_yes
# description: Yes rate for targets exactly 2 from the nearest example minus yes rate for targets exactly 1 away; observed > null means people favour same-parity neighbours over adjacent numbers more than the model's interval/parity mix predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m1, m2 = d == 1, d == 2
    if m1.sum() == 0 or m2.sum() == 0:
        return 0.0
    return float(y[m2].mean() - y[m1].mean())
