# name: single_example_yes_rate
# description: Yes rate on trials with exactly one example (non-example targets); observed > null means the model generalizes too narrowly from a single example, observed < null too broadly.
def test_statistic(df):
    k = (df["number_set"].astype(str).str.count(",") + 1).to_numpy()
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    notex = ~(ex == t[:, None]).any(axis=1)
    m = (k == 1) & notex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
