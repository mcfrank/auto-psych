# name: tens_sets_nonround_targets
# description: For sets of >= 2 examples that are all multiples of 10, yes-rate for targets that are not multiples of 10; observed > null means people accept non-round numbers for all-round sets more than the model's strong multiples-of-10 rule predicts (observed < null: they reject them more).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(axis=1)
    allten = np.where(np.isnan(ex), True, np.nan_to_num(ex) % 10 == 0).all(axis=1)
    mask = (n >= 2) & allten & (t % 10 != 0)
    return float(r[mask].mean()) if mask.any() else 0.0
