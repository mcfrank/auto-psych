# name: single_example_generalization_rate
# description: Yes rate on non-example targets for sets with exactly one example; observed above null means the model under-generalises from a single example (too narrow), below means it over-generalises.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(float).values
    n = (~np.isnan(ex)).sum(axis=1)
    m = (n == 1) & ~(ex == t[:, None]).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
