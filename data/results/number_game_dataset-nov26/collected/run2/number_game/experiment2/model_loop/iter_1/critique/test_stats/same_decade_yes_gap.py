# name: same_decade_yes_gap
# description: Yes-rate on non-example targets sharing a tens digit (same decade) with some example minus yes-rate on non-example targets sharing no decade; observed below null means people avoid same-decade targets more than the model predicts, above means decade similarity the model lacks.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    sd = np.any(np.floor(E / 10) == np.floor(t / 10)[:, None], axis=1)
    a = sd & (~inset)
    b = (~sd) & (~inset)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
