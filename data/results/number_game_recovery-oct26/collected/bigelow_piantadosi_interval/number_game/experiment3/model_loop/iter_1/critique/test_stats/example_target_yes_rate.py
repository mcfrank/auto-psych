# name: example_target_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below null_mean means the model over-predicts endorsing the examples themselves (people are less at ceiling), above means it under-predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None]
    is_ex = (ex == t).any(axis=1)
    if is_ex.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[is_ex].mean())
