# name: single_example_generalization
# description: Yes-rate on non-example targets for single-example sets; observed above the null means people generalize more broadly from one example than the model's size-principle posterior predicts, below means more narrowly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    ex = pd.to_numeric(s.where(single, "-1"), errors="coerce").to_numpy()
    t = df["target"].astype(float).to_numpy()
    m = single.to_numpy() & (ex != t)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
