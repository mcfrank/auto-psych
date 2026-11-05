# name: last_digit_match_effect
# description: Yes-rate for targets sharing a last digit with at least one example minus yes-rate for targets sharing none; observed > null means humans use surface digit similarity more than the model's ends-in-d rules predict, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    match = np.any(np.mod(ex, 10) == np.mod(t, 10)[:, None], axis=1)
    y = df["rating"].to_numpy().astype(float)
    if match.all() or (~match).all():
        return 0.0
    return float(y[match].mean() - y[~match].mean())
