# name: target_in_set_yes_rate
# description: Yes rate when the target is one of the shown examples (0.5 if no such trials); observed below null means people reject already-seen examples more than the model (which says near-certain yes) predicts.
def test_statistic(df):
    t = df["target"].astype(int).astype(str)
    s = "," + df["number_set"].astype(str).str.replace(" ", "") + ","
    m = np.array([("," + a + ",") in b for a, b in zip(t, s)])
    if m.sum() == 0:
        return 0.5
    return float(df["rating"].to_numpy()[m].mean())
