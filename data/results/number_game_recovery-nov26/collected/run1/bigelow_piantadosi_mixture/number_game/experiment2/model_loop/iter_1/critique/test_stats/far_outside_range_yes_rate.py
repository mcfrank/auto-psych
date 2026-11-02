# name: far_outside_range_yes_rate
# description: Yes rate for targets more than 30 below the smallest or above the largest example; observed above null means people say yes to far-away numbers (lapses / broad rules) more than the model predicts, below means the model over-generalises far.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(float).to_numpy()
    mn = s.min(axis=1).to_numpy(); mx = s.max(axis=1).to_numpy()
    far = (t < mn - 30) | (t > mx + 30)
    r = df["rating"].astype(float).to_numpy()
    if far.sum() == 0:
        return 0.0
    return float(r[far].mean())
