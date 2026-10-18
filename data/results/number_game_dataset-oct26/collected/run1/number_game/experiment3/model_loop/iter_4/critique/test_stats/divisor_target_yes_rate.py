# name: divisor_target_yes_rate
# description: Yes-rate on targets (>=2) that are proper divisors of some example, excluding examples and targets sharing any example's final digit; observed > null means people generalise "downward" to factors of the examples (e.g. 16 -> 8, 100 -> 25) more than the model, which only credits multiples/counting up; observed < null the reverse.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    Xf = np.where(valid, X, 1e9)
    is_div = valid & (t >= 2) & (t < Xf) & (np.mod(Xf, t) == 0)
    same_digit = valid & (np.mod(Xf, 10) == np.mod(t, 10))
    is_ex = valid & (Xf == t)
    m = is_div.any(1) & ~same_digit.any(1) & ~is_ex.any(1)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
