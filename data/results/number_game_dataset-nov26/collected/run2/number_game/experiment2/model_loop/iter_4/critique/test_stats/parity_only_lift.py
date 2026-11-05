# name: parity_only_lift
# description: In multi-example sets whose examples all share parity, yes-rate for same-parity targets minus opposite-parity targets, restricted to targets sharing no final digit with any example and not a multiple of the smallest example; observed > null means the model under-weights the parity property.
def test_statistic(df):
    P = df["number_set"].astype(str).str.split(",", expand=True)
    X = np.column_stack([pd.to_numeric(P[c], errors="coerce").to_numpy(float) for c in P.columns])
    n = (~np.isnan(X)).sum(1)
    t = df["target"].astype(int).to_numpy().astype(float); r = df["rating"].to_numpy(float)
    lo = np.nanmin(X, 1)
    pmax = np.nanmax(X % 2, 1); pmin = np.nanmin(X % 2, 1)
    D = np.where(np.isnan(X), -1, X % 10)
    digit = (D == (t % 10)[:, None]).any(1)
    base = (n >= 2) & (pmax == pmin) & ~digit & ((t % lo) != 0)
    same = base & ((t % 2) == pmin)
    diff = base & ((t % 2) != pmin)
    if same.sum() < 5 or diff.sum() < 5:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
