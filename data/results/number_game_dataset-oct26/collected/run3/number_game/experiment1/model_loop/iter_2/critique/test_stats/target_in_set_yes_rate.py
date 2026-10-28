# name: target_in_set_yes_rate
# description: Yes-rate on trials whose target is itself one of the shown examples; the model forces this to ~1, so observed below null means the model over-produces yes for already-seen examples (people are noisier/less certain than any concept-membership account).
def test_statistic(df):
    ns = "," + df["number_set"].astype(str).str.replace(" ", "", regex=False) + ","
    tg = "," + df["target"].astype(int).astype(str) + ","
    inset = np.array([b in a for a, b in zip(ns.to_numpy(), tg.to_numpy())])
    if not inset.any():
        return 1.0
    return float(df["rating"].to_numpy(float)[inset].mean())
