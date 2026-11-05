# name: single_set_breadth_dispersion
# description: SD across one-example sets of each set's overall yes-rate; observed > null means people's breadth of generalisation varies by the single example more than the model predicts (model treats one-example sets too uniformly).
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    s = df[n.to_numpy() == 1]
    if len(s) == 0:
        return 0.0
    m = s.groupby("number_set")["rating"].mean().to_numpy(float)
    if len(m) < 2:
        return 0.0
    return float(np.std(m))
