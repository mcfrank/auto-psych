# name: example_target_yes_rate
# description: Yes-rate among trials whose target is one of the shown examples (sets of >=2); observed below null means the model over-produces "yes" to the examples themselves (people sometimes reject a shown example).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    n = sets.str.count(",") + 1
    t = df["target"].astype(int).astype(str)
    padded = "," + sets.str.replace(" ", "", regex=False) + ","
    is_ex = pd.Series([("," + a + ",") in b for a, b in zip(t.values, padded.values)], index=df.index)
    m = is_ex & (n >= 2)
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].astype(float).mean())
