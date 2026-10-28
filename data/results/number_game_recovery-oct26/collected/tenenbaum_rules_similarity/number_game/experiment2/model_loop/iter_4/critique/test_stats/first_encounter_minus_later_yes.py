# name: first_encounter_minus_later_yes
# description: Within each session (inferred from trial_index resets), yes-rate on non-example targets for the first two trials on which a given example set appears minus the yes-rate on that set's later trials in the session; observed above null means people narrow a set's concept with repeated exposure (below null: broaden), an exposure dynamic the static model lacks.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    sess = np.concatenate([[0], np.cumsum(np.diff(ti) < 0)])
    key = pd.Series(sess.astype(str)) + "|" + df["number_set"].astype(str).reset_index(drop=True)
    occ = key.groupby(key).cumcount().to_numpy()
    t = df["target"].astype(int).astype(str).to_numpy()
    s = ("," + df["number_set"].astype(str).str.replace(" ", "") + ",").to_numpy()
    non_ex = np.array([("," + a + ",") not in b for a, b in zip(t, s)])
    r = df["rating"].to_numpy().astype(float)
    early = non_ex & (occ < 2)
    late = non_ex & (occ >= 2)
    if early.sum() == 0 or late.sum() == 0:
        return 0.0
    return float(r[early].mean() - r[late].mean())
