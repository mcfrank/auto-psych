# name: small_vs_large_max_overshoot_gap
# description: Yes rate for targets 1-12 above the largest example when that maximum is at most 40, minus the same when the maximum is 55-88; observed below null means a fixed absolute overshoot is penalised even more above small maxima than the model's log-ratio criterion implies, above null means the ratio scaling overstates the small-vs-large asymmetry.
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).astype(float)
    mx = parts.max(axis=1).values
    t = df["target"].astype(float).values
    ov = t - mx
    near = (ov >= 1) & (ov <= 12)
    r = df["rating"].values.astype(float)
    a = near & (mx <= 40)
    b = near & (mx >= 55) & (mx <= 88)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
