# name: shared_last_digit_yes
# description: Yes rate for non-example targets whose last digit matches the shared last digit of a >= 2-example set (all examples end in the same digit); observed > null means people endorse "ends in d" rule extensions more than the model, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    ld = np.where(np.isnan(ex), np.nan, np.mod(ex, 10))
    same = (np.nanmax(ld, axis=1) == np.nanmin(ld, axis=1)) & (n >= 2)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = same & (np.mod(t, 10) == np.nanmax(ld, axis=1)) & (d > 0)
    return float(y[m].mean()) if m.sum() else 0.0
