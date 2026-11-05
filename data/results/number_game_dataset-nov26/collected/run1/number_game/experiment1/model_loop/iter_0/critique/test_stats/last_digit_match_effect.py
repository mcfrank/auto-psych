# name: last_digit_match_effect
# description: Yes-rate for targets whose last digit equals the last digit of at least one example minus yes-rate for targets matching none; observed above null_mean means people generalize by shared final digit (even partially/for mixed sets) more than the model, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    match = np.any(np.mod(ex, 10) == (t % 10)[:, None], axis=1)
    if match.sum() == 0 or (~match).sum() == 0:
        return 0.0
    return float(y[match].mean() - y[~match].mean())
