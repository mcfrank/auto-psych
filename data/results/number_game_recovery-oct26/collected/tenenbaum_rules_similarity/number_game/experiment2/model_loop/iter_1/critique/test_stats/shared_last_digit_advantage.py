# name: shared_last_digit_advantage
# description: Among non-example targets with the same parity as at least one example, yes rate for targets sharing a final digit with some example minus yes rate for those that do not; observed above null means the model under-weights final-digit similarity, below null means it over-weights it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    is_ex = np.any(ex == t[:, None], axis=1)
    same_par = np.any(np.mod(ex, 2) == np.mod(t, 2)[:, None], axis=1)
    share = np.any(np.mod(ex, 10) == np.mod(t, 10)[:, None], axis=1)
    y = df["rating"].to_numpy(dtype=float)
    base = (~is_ex) & same_par
    a = base & share
    b = base & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
