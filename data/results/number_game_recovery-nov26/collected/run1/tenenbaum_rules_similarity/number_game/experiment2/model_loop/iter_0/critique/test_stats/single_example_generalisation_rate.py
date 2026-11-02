# name: single_example_generalisation_rate
# description: Yes rate for single-example sets on targets different from the example; observed above null means the model under-generalises from one example (too narrow), below null means it over-generalises.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",")
    ex = pd.to_numeric(s.str.split(",").str[0], errors="coerce").to_numpy(dtype=float)
    m = one.to_numpy() & (ex != df["target"].astype(float).to_numpy())
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
