# name: single_digit_targets_for_high_sets
# description: For sets whose smallest example is >= 20, yes-rate for single-digit targets (1-9); observed > null means people accept small one-digit numbers for high-valued sets more than the model's rules/intervals predict (observed < null: they treat one-digit numbers as a separate category and reject them more than the model).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    mn = np.nanmin(ex, axis=1)
    mask = (mn >= 20) & (t <= 9)
    return float(r[mask].mean()) if mask.any() else 0.0
