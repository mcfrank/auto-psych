# name: example_target_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below the replicate mean means the model's guessing ceiling is too high (people reject examples more), above means people endorse examples more than the lapse allows.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).to_numpy()[:, None]
    m = (ex.to_numpy() == t).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
