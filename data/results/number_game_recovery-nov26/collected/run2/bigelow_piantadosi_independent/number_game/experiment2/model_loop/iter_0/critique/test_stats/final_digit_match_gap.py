# name: final_digit_match_gap
# description: In sets of >=2 examples all sharing a final digit, yes rate for targets with that final digit minus targets without; observed above null means people pick up final-digit rules more strongly than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    fd = np.where(np.isnan(ex), np.nan, ex % 10)
    same = (n >= 2) & (np.nanmax(fd, axis=1) == np.nanmin(fd, axis=1))
    match = (t % 10) == np.nanmin(fd, axis=1)
    return rate(same & match) - rate(same & ~match)
