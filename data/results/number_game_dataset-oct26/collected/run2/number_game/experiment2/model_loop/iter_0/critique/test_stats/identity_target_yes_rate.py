# name: identity_target_yes_rate
# description: Yes-rate on trials where the target is itself one of the shown examples; observed below the model's null means the model over-endorses already-seen examples (people reject some, e.g. 90/100 given 80,70,90,100), above means it under-endorses them.
def test_statistic(df):
    pairs = df[["number_set", "target"]].drop_duplicates()
    ident = {(s, t): str(int(t)) in str(s).split(",") for s, t in zip(pairs["number_set"], pairs["target"])}
    key = pd.Series(list(zip(df["number_set"], df["target"])), index=df.index)
    m = key.map(ident).astype(bool).values
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
