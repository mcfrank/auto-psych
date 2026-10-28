# name: example_target_yes_rate
# description: Yes-rate on trials whose target is itself one of the shown examples; observed above the model's null means the model under-produces yeses to the examples (its noise/gain/bias leak too much belief away from seen numbers), below means it over-produces them.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    is_ex = (ex == t[:, None]).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    if is_ex.sum() == 0:
        return 0.0
    return float(r[is_ex].mean())
