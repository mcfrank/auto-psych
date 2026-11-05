# name: shared_decade_bonus
# description: Among 2+-example sets and targets whose final digit matches no example, yes rate for targets sharing a tens digit (decade) with some example minus yes rate for targets sharing no decade; observed > null means people use a same-decade similarity the model lacks.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1).values
    E = ex.values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    dec_match = np.nansum((np.floor(E / 10) == np.floor(t / 10)[:, None]) & ~np.isnan(E), axis=1) > 0
    ld_match = np.nansum((np.mod(E, 10) == np.mod(t, 10)[:, None]) & ~np.isnan(E), axis=1) > 0
    base = (n >= 2) & (~ld_match)
    r = df["rating"].values.astype(float)
    a = base & dec_match; b = base & (~dec_match)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
