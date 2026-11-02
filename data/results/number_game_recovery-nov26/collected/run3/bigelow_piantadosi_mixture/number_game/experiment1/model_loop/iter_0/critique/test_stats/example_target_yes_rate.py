# name: example_target_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed < null means the model over-produces "yes" for already-seen examples (people are not certain about given members).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    m = (ex == t[:, None]).any(axis=1)
    if m.sum() == 0:
        return 1.0
    return float(df["rating"].to_numpy()[m].mean())
