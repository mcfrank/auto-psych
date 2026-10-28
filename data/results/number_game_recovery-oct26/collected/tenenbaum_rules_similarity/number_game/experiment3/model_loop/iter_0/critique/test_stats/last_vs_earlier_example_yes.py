# name: last_vs_earlier_example_yes
# description: Yes-rate when the target is the LAST-shown example minus yes-rate when it is an earlier-shown example (sets of >=2); observed below null means people reject the final example more than the model (which treats all examples alike, or favours a last max) predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.replace(" ", "", regex=False)
    n = sets.str.count(",") + 1
    t = df["target"].astype(int).values
    last = sets.str.split(",").str[-1].astype(int).values
    padded = ("," + sets + ",").values
    is_ex = np.array([("," + str(a) + ",") in b for a, b in zip(t, padded)])
    multi = (n >= 2).values
    r = df["rating"].astype(float).values
    a = multi & is_ex & (t == last)
    b = multi & is_ex & (t != last)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
