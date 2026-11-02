# name: adjacent_to_example_yes_rate
# description: On 2+ example sets, yes rate for non-example targets exactly 1 away from some example minus those 2-5 away from the nearest example; observed above null means immediate numerical neighbours are accepted more than the model's rule/interval mixture predicts (local similarity), below null means neighbours are rejected (e.g. parity breaks) more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(ex)).sum(1)
    t = df["target"].astype(float).values
    dist = np.nanmin(np.abs(ex - t[:, None]), 1)
    r = df["rating"].astype(float).values
    m = n >= 2
    a = r[m & (dist == 1)]; b = r[m & (dist >= 2) & (dist <= 5)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
