# name: target_is_example_yes_rate
# description: Yes-rate on trials whose target is one of the shown examples; observed < model means people reject already-shown examples far more often than a concept-membership model (which puts them in nearly every posterior concept) predicts, observed > model means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[:, None]
    m = (ex == t).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(dtype=float)[m].mean())
