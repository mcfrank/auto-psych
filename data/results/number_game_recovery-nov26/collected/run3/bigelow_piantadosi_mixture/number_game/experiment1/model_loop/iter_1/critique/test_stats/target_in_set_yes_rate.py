# name: target_in_set_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed < null means the model over-predicts near-certain "yes" for already-seen numbers (people lapse/deny more).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float).values
    m = (sets.values == t[:, None]).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
