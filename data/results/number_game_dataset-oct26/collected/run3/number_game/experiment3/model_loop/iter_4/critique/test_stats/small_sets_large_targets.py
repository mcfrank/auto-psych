# name: small_sets_large_targets
# description: For sets of >= 2 examples that are all <= 10, yes-rate for targets > 20; observed > null means people extend small-number sets to large numbers (via rules like odd/multiples) more than the model predicts (observed < null: they stay confined to small numbers more than the model).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(axis=1)
    mx = np.nanmax(ex, axis=1)
    mask = (n >= 2) & (mx <= 10) & (t > 20)
    return float(r[mask].mean()) if mask.any() else 0.0
