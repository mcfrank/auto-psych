# name: set_size_nonmember_slope
# description: OLS slope of the yes response on number of examples (1-4) among non-example targets more than 10 away from every example; a more negative observed slope than null means people tighten generalization with more examples more strongly (stronger size principle) than the model, a less negative one means weaker.
def test_statistic(df):
    n = (df["number_set"].str.count(",") + 1).to_numpy().astype(float)
    t = df["target"].astype(int).to_numpy()
    exl = df["number_set"].map(lambda s: [int(x) for x in s.split(",")]).to_numpy()
    dist = np.array([min(abs(tt - e) for e in ex) for tt, ex in zip(t, exl)])
    m = dist > 10
    x = n[m]
    y = df["rating"].to_numpy().astype(float)[m]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])
