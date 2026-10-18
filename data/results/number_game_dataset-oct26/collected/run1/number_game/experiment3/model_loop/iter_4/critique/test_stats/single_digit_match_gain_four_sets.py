# name: single_digit_match_gain_four_sets
# description: In 4-example sets whose examples do NOT all share a final digit, yes-rate on non-example targets sharing the final digit of exactly one example minus on targets sharing none; observed > null means a single matching exemplar boosts acceptance more than the model's averaged (1/4-weighted) similarity allows (max-like rather than mean-like exemplar matching); observed < null the reverse.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    n = valid.sum(1)
    d = np.where(valid, np.mod(np.nan_to_num(X), 10), -1)
    dmin = np.where(valid, d, 99).min(1); dmax = np.where(valid, d, -99).max(1)
    uniform = dmin == dmax
    cnt = (valid & (d == np.mod(t, 10))).sum(1)
    is_ex = (valid & (X == t)).any(1)
    base = (n == 4) & ~uniform & ~is_ex
    a = base & (cnt == 1); b = base & (cnt == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
