# name: shown_example_endorsement
# description: Yes-rate on trials whose target is one of the shown examples, in sets of 2-4 examples; observed below the model's null means people reject shown examples (esp. outliers) more than the model, which nearly always includes them in the concept.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    m = is_ex & (n >= 2)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
