# name: shown_example_target_yes
# description: Yes rate on trials whose target is one of the set's own examples; observed below null means the model over-predicts endorsing the shown examples (people distrust examples more than its 10% stray-example noise allows), above null means it under-predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    tin = (ex == t[:, None]).any(axis=1)
    if not tin.any():
        return 0.0
    return float(df["rating"].to_numpy()[tin].mean())
