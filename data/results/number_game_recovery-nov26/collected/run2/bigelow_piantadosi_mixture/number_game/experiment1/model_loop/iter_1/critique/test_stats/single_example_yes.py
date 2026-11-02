# name: single_example_yes
# description: Yes rate on single-example sets with target different from the example (breadth of generalization from one number); observed above null means people generalize more broadly from one example than the model, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    ex = pd.to_numeric(s.where(single, "-1"), errors="coerce").fillna(-1).astype(int).to_numpy()
    m = single.to_numpy() & (ex != df["target"].astype(int).to_numpy())
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
