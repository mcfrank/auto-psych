# name: last_digit_match_far
# description: Among non-example targets more than 10 from every example, yes rate for targets sharing a last digit with at least one example minus targets that share none; the model credits this only via ends-in-d rules, so observed above null_mean means people use last-digit similarity more than the model predicts, below means less.
def test_statistic(df):
    ex = pd.DataFrame(df["number_set"].astype(str).str.split(",").tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(y[a].mean() - y[b].mean())
    share = np.any(np.mod(ex, 10) == np.mod(t, 10)[:, None], axis=1)
    base = dist > 10
    return diff(base & share, base & ~share)
