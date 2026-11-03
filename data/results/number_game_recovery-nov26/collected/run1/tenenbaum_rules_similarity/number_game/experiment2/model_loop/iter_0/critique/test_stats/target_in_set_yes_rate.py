# name: target_in_set_yes_rate
# description: Yes rate on trials where the target is itself one of the shown examples; observed below null means the model over-predicts certain "yes" for already-seen numbers (people are less deterministic / more lapse-prone there).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    m = (ex == t[:, None]).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
