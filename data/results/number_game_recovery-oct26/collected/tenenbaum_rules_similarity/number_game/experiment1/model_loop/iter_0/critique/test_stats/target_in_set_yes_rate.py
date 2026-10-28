# name: target_in_set_yes_rate
# description: Proportion of "yes" when the target is one of the shown examples; observed below the model's null means the model over-predicts endorsement of already-seen examples (humans lapse/hedge even on members), above means it under-predicts it.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).astype(str)
    padded = "," + s + ","
    inset = np.array([p.find("," + x + ",") >= 0 for p, x in zip(padded.values, t.values)])
    if inset.sum() == 0:
        return 0.0
    return float(df["rating"].values[inset].mean())
