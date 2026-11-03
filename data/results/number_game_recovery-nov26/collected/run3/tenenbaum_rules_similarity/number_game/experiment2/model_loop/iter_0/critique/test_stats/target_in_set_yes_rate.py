# name: target_in_set_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed < null means people sometimes reject an already-shown example (lapses/random responding) that the lapse-free model, predicting near-certain yes, does not produce.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    in_set = np.any(E == t[:, None], axis=1)
    r = df["rating"].to_numpy().astype(float)
    if in_set.sum() == 0:
        return 0.0
    return float(r[in_set].mean())
