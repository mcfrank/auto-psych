# name: near_example_proximity_lift
# description: In multi-example sets, yes-rate for targets within +-2 of some example (not equal, not sharing any example's final digit) minus yes-rate for other non-digit-sharing targets inside the range; observed > null means the model misses local numeric-neighbourhood generalisation in multi-example sets.
def test_statistic(df):
    P = df["number_set"].astype(str).str.split(",", expand=True)
    X = np.column_stack([pd.to_numeric(P[c], errors="coerce").to_numpy(float) for c in P.columns])
    n = (~np.isnan(X)).sum(1)
    t = df["target"].astype(int).to_numpy().astype(float); r = df["rating"].to_numpy(float)
    lo = np.nanmin(X, 1); hi = np.nanmax(X, 1)
    dist = np.nanmin(np.abs(np.where(np.isnan(X), 1e9, X) - t[:, None]), 1)
    D = np.where(np.isnan(X), -1, X % 10)
    digit = (D == (t % 10)[:, None]).any(1)
    base = (n >= 2) & ~digit & (t >= lo - 2) & (t <= hi + 2)
    near = base & (dist >= 1) & (dist <= 2)
    far = base & (dist > 2)
    if near.sum() < 5 or far.sum() < 5:
        return 0.0
    return float(r[near].mean() - r[far].mean())
