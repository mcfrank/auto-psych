# name: high_minus_low_multi_example
# description: For sets with 2+ examples, non-example-target yes rate for targets > 50 minus targets <= 50; observed above null means people favour large targets more than the model (a magnitude bias the model lacks), below = model over-favours large targets.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n >= 2]
    t = sub["target"].astype(int)
    padded = "," + sub["number_set"].str.replace(" ", "", regex=False) + ","
    is_ex = pd.Series([("," + str(a) + ",") in b for a, b in zip(t.values, padded.values)], index=sub.index)
    r = sub["rating"]
    hi = (~is_ex) & (t > 50); lo = (~is_ex) & (t <= 50)
    if hi.sum() == 0 or lo.sum() == 0:
        return 0.0
    return float(r[hi].mean() - r[lo].mean())
