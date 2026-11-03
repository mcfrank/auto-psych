# name: example_target_yes_rate
# description: Yes-rate on trials whose target is itself one of the shown examples; observed below null means the model over-endorses already-seen examples (people reject or doubt literal examples more than an inferred-noise concept learner predicts), above null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None]
    m = (ex == t).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    return float(r[m].mean()) if m.sum() > 0 else 0.0
