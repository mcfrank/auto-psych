# name: in_set_target_yes_rate
# description: Yes rate when the target is itself one of the shown examples; observed below null means the model over-endorses example members (people reject listed examples more than the model's near-certain membership allows).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    return m(inset)
