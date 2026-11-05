# name: seed_multiple_out_of_range_rate
# description: In multi-example sets where every example is a multiple of the smallest one (>=2), yes-rate for multiples of that seed lying ABOVE the examples' max; observed > null means people extend the multiple rule beyond the shown range more than the model's magnitude band allows.
def test_statistic(df):
    P = df["number_set"].astype(str).str.split(",", expand=True)
    X = np.column_stack([pd.to_numeric(P[c], errors="coerce").to_numpy(float) for c in P.columns])
    n = (~np.isnan(X)).sum(1)
    t = df["target"].astype(int).to_numpy().astype(float); r = df["rating"].to_numpy(float)
    lo = np.nanmin(X, 1); hi = np.nanmax(X, 1)
    lo_s = np.where(lo < 2, 1.0, lo)
    rem = np.where(np.isnan(X), 0, X % lo_s[:, None])
    allmult = (rem == 0).all(1) & (lo >= 2)
    sel = (n >= 2) & allmult & ((t % lo_s) == 0) & (t > hi)
    if sel.sum() < 5:
        return 0.0
    return float(r[sel].mean())
