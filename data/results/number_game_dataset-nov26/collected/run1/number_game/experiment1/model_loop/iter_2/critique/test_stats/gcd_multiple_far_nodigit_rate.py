# name: gcd_multiple_far_nodigit_rate
# description: Among sets of >=2 examples whose GCD is >=3, yes-rate for targets that are multiples of the GCD, lie >10 from every example and share no final digit with any example, minus yes-rate for comparable non-multiples; observed > null means people generalize by arithmetic rule (multiples) beyond what magnitude/digit similarity predicts.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    n = (~np.isnan(ex)).sum(1)
    exi = np.nan_to_num(ex, nan=0).astype(int)
    g = np.zeros(len(t), dtype=int)
    for j in range(exi.shape[1]):
        g = np.gcd(g, exi[:, j])
    ti = t.astype(int)
    d = np.nanmin(np.abs(ex - t[:, None]), 1)
    dig = np.any((np.mod(exi, 10) == np.mod(ti, 10)[:, None]) & ~np.isnan(ex), 1)
    base = (n >= 2) & (g >= 3) & (d > 10) & ~dig
    mult = base & (np.mod(ti, np.maximum(g, 1)) == 0)
    non = base & (np.mod(ti, np.maximum(g, 1)) != 0)
    a = y[mult].mean() if mult.any() else 0.0
    b = y[non].mean() if non.any() else 0.0
    return float(a - b)
