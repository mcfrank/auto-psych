# name: inrange_nonshared_rate
# description: In multi-example sets, yes-rate for targets strictly inside the examples' [min,max] range that share neither final digit, parity nor seed-multiple with the examples, minus the same for targets outside the range; observed > null means the model under-produces pure magnitude ("in the range") generalisation.
def test_statistic(df):
    P = df["number_set"].astype(str).str.split(",", expand=True)
    X = np.column_stack([pd.to_numeric(P[c], errors="coerce").to_numpy(float) for c in P.columns])
    n = (~np.isnan(X)).sum(1)
    t = df["target"].astype(int).to_numpy().astype(float); r = df["rating"].to_numpy(float)
    lo = np.nanmin(X, 1); hi = np.nanmax(X, 1)
    D = np.where(np.isnan(X), -1, X % 10)
    digit = (D == (t % 10)[:, None]).any(1)
    par = np.where(np.isnan(X), -1, X % 2)
    allpar_same = np.nanmax(X % 2, 1) == np.nanmin(X % 2, 1)
    parity = allpar_same & ((t % 2) == np.nanmin(X % 2, 1))
    seedmult = (t % lo) == 0
    clean = (n >= 2) & ~digit & ~parity & ~seedmult
    inr = clean & (t > lo) & (t < hi)
    outr = clean & ((t < lo) | (t > hi))
    if inr.sum() < 5 or outr.sum() < 5:
        return 0.0
    return float(r[inr].mean() - r[outr].mean())
