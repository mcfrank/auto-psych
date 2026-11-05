# name: parity_contrast_far_nodigit
# description: For sets whose examples all share parity (>=2 examples), yes-rate for same-parity targets minus opposite-parity targets, restricted to targets >10 from every example and sharing no final digit; observed > null means people use parity (even/odd) as a generalization dimension the model lacks.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    return ex, t, y

def test_statistic(df):
    ex, t, y = _parse(df)
    n = (~np.isnan(ex)).sum(1)
    exi = np.nan_to_num(ex, nan=-1).astype(int)
    valid = ~np.isnan(ex)
    par = np.mod(exi, 2)
    alle = np.all((par == 0) | ~valid, 1); allo = np.all((par == 1) | ~valid, 1)
    ti = t.astype(int)
    d = np.nanmin(np.abs(ex - t[:, None]), 1)
    dig = np.any((np.mod(exi, 10) == np.mod(ti, 10)[:, None]) & valid, 1)
    base = (n >= 2) & (alle | allo) & (d > 10) & ~dig
    setpar = np.where(alle, 0, 1)
    same = base & (np.mod(ti, 2) == setpar)
    opp = base & (np.mod(ti, 2) != setpar)
    a = y[same].mean() if same.any() else 0.0
    b = y[opp].mean() if opp.any() else 0.0
    return float(a - b)
