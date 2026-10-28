# name: shared_last_digit_contrast
# description: Among multi-example sets whose examples all share a last digit, yes rate for non-example targets with that last digit minus yes rate for targets without it; observed above the replicate mean means people follow the last-digit rule more sharply than the model, below means less sharply.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.mod(ex, 10)
    first = d[:, 0]
    same = np.all(np.isnan(d) | (d == first[:, None]), axis=1) & (n >= 2)
    isex = (ex == t[:, None]).any(axis=1)
    match = np.mod(t, 10) == first
    r = df["rating"].to_numpy().astype(float)
    a = same & ~isex & match
    b = same & ~isex & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
