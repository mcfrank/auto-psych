# name: round_number_target_yes
# description: Yes rate for targets that are multiples of 10 on sets containing no multiple-of-10 example and with the target 3+ from every example; observed > null means people endorse round numbers as a salient default beyond the model, observed < null less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    ex10 = np.any(np.nan_to_num(ex, nan=1.0) % 10 == 0, axis=1)
    return rate((t % 10 == 0) & ~ex10 & (dist >= 3))
