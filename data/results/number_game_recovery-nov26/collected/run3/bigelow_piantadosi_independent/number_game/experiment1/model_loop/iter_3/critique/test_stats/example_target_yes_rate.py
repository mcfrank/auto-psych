# name: example_target_yes_rate
# description: Yes rate for targets identical to one of the shown examples; observed < null means people reject already-seen examples more often than the model (which predicts near-certain yes minus lapse), observed > null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    m = ex.eq(t, axis=0).any(axis=1).to_numpy()
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
