# name: target_in_set_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below the model's null means the model is over-confident (lacks lapses/doubt) on exact example targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(float).values[:, None]
    m = (ex == t).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
