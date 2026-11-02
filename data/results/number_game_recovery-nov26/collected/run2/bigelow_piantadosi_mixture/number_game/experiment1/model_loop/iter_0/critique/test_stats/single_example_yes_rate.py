# name: single_example_yes_rate
# description: Yes rate on trials whose set has exactly one example, excluding target equal to the example (0.5 if none); observed above/below null means the model under/over-generalises from a single example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).to_numpy()
    ex1 = pd.to_numeric(s.str.split(",").str[0], errors="coerce").to_numpy()
    tgt = df["target"].astype(int).to_numpy()
    m = one & (ex1 != tgt)
    if m.sum() == 0:
        return 0.5
    return float(df["rating"].to_numpy()[m].mean())
