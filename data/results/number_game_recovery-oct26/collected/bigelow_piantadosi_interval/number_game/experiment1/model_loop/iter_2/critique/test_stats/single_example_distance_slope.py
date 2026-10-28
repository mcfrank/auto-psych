# name: single_example_distance_slope
# description: Least-squares slope of yes (0/1) on |target - example| / 10 for non-example targets of single-example sets; observed more negative than null means people's generalization from one example falls off with numerical distance more steeply than the model's interval/rule mixture predicts, less negative means flatter.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",", regex=False).to_numpy()
    t = df["target"].astype(int).to_numpy()
    e = pd.to_numeric(sets.where(single, "0"), errors="coerce").fillna(0).astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = single & (t != e)
    if sel.sum() < 3:
        return 0.0
    x = np.abs(t[sel] - e[sel]) / 10.0
    y = r[sel]
    vx = np.var(x)
    if vx <= 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
