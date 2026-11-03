# name: target_is_example_yes_rate
# description: Yes rate on trials whose target equals one of the shown examples; observed below null_mean means the model is too confident that examples belong to the concept (people need a lapse/noise floor at the top).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    m = np.any(E == t[:, None], axis=1)
    return float(r[m].mean()) if m.any() else 0.0
