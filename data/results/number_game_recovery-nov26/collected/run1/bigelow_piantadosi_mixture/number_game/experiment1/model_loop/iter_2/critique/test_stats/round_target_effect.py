# name: round_target_effect
# description: Among sets not all multiples of 10, yes-rate for targets that are multiples of 10 minus other targets, both lying outside the examples' range; observed > null means people over-accept round numbers beyond what the model's rule mixture predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    allten = np.all(np.isnan(ex) | (np.mod(ex, 10) == 0), axis=1)
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    rt = np.mod(t, 10) == 0
    m1 = ~allten & out & rt; m0 = ~allten & out & ~rt
    a = r[m1].mean() if m1.any() else 0.0; b = r[m0].mean() if m0.any() else 0.0
    return float(a - b)
