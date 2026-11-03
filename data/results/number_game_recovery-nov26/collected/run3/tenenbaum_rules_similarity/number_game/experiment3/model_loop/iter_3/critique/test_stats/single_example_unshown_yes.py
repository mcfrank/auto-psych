# name: single_example_unshown_yes
# description: Yes rate for unshown targets on single-example sets; observed above null means people generalize more broadly from one example than the model, below means more conservatively.
def test_statistic(df):
    n = df["number_set"].str.count(",").to_numpy() + 1
    t = df["target"].astype(int).to_numpy()
    first = pd.to_numeric(df["number_set"].str.split(",").str[0]).to_numpy()
    m = (n == 1) & (t != first)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(dtype=float)[m].mean())
