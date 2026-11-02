# name: target_100_yes
# description: Yes rate for target 100 across all sets not containing 100; observed above null means people over-accept the salient endpoint 100 relative to the model.
def test_statistic(df):
    t = df["target"].astype(int).values
    has = df["number_set"].astype(str).str.contains(r"(^|,)100($|,)").values
    m = (t == 100) & ~has
    return float(df["rating"].values[m].mean()) if m.any() else 0.0
