# name: singleton_near_generalization
# description: For single-example sets, yes-rate to non-example targets within 10 of the example; observed above null means people generalise to neighbours of a lone example more broadly than the model predicts, below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.replace(" ", "", regex=False)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).astype(int).values
    t = df["target"].astype(int).values
    d = np.abs(t - x)
    m = single.values & (d >= 1) & (d <= 10)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
