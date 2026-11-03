# name: shared_last_digit_set_mismatch_yes
# description: Among sets of 2+ examples that all share the same final digit (e.g. 97,77,87,47; 7,67; 81,71,21,31), yes rate for targets with a different final digit; observed below null means people reject off-rule targets more completely than the model (model's lapse/range fallback too generous), above means they accept them more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    e = np.where(np.isnan(ex), ex[:, :1], ex)
    same = (e % 10 == (e[:, :1] % 10)).all(1)
    sel = (k >= 2) & same & (t % 10 != e[:, 0] % 10)
    return m(sel)
