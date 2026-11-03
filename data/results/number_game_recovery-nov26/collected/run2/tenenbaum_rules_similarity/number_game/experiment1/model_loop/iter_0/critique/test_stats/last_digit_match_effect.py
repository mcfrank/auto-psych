# name: last_digit_match_effect
# description: Among non-member targets, yes-rate when the target shares its last digit with at least one example minus when it shares with none; observed above null means humans use partial digit similarity more than the model's all-examples 'ends in d' rules allow, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    dm = np.any(valid & (np.mod(ex, 10) == np.mod(t, 10)), axis=1)
    a = (~member) & dm; b = (~member) & ~dm
    if not a.any() or not b.any(): return 0.0
    return float(r[a].mean() - r[b].mean())
