# name: member_target_yes_rate
# description: Yes-rate on trials whose target is one of the shown examples; observed below null means the model over-predicts yes for examples (humans lapse/say no to shown members far more than the model allows).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    return float(r[member].mean()) if member.any() else 0.0
