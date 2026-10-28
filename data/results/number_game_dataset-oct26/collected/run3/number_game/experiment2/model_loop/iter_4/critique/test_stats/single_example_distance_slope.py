# name: single_example_distance_slope
# description: Among one-example sets, OLS slope of yes (0/1) on |target - example|/100; a more negative observed value than null means people generalise more by numeric proximity from one example than the model predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    m = (s.str.count(",") == 0).values
    if m.sum() < 3:
        return 0.0
    ex = pd.to_numeric(s[m], errors="coerce").fillna(0).values
    t = df["target"].astype(float).values[m]
    r = df["rating"].astype(float).values[m]
    x = np.abs(t - ex) / 100.0
    v = x.var()
    if v <= 0:
        return 0.0
    return float(((x - x.mean()) * (r - r.mean())).mean() / v)
