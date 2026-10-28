# name: in_set_target_yes_rate
# description: Yes-rate on trials whose target is one of the shown examples; observed below null means people miss/lapse on trivially consistent targets that the lapse-free model answers "yes" to almost surely.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).astype(str)
    padded = "," + s.str.replace(" ", "", regex=False) + ","
    inset = np.array([("," + b + ",") in a for a, b in zip(padded.values, t.values)])
    if inset.sum() == 0:
        return 0.0
    return float(df["rating"].values[inset].mean())
