# name: single_example_generalization_rate
# description: Yes rate on single-example sets for targets different from the example; observed > null means the model generalizes too narrowly from one example (too-strong size principle / too-narrow intervals), observed < null means it generalizes too broadly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    t = df["target"].astype(int).to_numpy()
    first = pd.to_numeric(s.str.split(",").str[0]).to_numpy()
    m = single & (t != first)
    r = df["rating"].to_numpy().astype(float)
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
