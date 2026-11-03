# name: leading_digit_match_gap
# description: Among sets with 2+ examples, for two-digit targets whose final digit matches no example, yes rate when the target's tens digit equals some example's tens digit minus yes rate when it equals none; observed above null means people generalise along shared leading digits (same decade) more than the model's ranges and contains-digit rules predict, below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(parts)
    ex_tens = np.where(valid & (parts >= 10) & (parts < 100), np.floor(parts / 10), -1)
    ex_ones = np.where(valid, np.mod(parts, 10), -1)
    t = df["target"].astype(int).to_numpy()
    tt = (t // 10).astype(float)
    to = (t % 10).astype(float)
    tens_match = (ex_tens == tt[:, None]).any(axis=1)
    ones_match = (ex_ones == to[:, None]).any(axis=1)
    member = (parts == t[:, None]).any(axis=1)
    multi = (s.str.count(",") >= 1).to_numpy()
    base = multi & (t >= 10) & (t < 100) & ~ones_match & ~member
    r = df["rating"].astype(float).to_numpy()
    a = base & tens_match
    b = base & ~tens_match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
