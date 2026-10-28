# name: single_example_generalization_rate
# description: Yes rate for non-example targets when only one example is shown; observed above the replicate mean means people generalize more broadly from a single example than the model's size-principle posterior, below means more narrowly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",").to_numpy()
    num = pd.to_numeric(s.str.split(",").str[0], errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy()
    m = one & (num != t)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
