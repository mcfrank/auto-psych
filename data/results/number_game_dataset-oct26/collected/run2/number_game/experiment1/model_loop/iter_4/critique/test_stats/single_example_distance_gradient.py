# name: single_example_distance_gradient
# description: In 1-example sets, yes rate for targets within 10 of the example (excluding the example itself) minus yes rate for targets more than 30 away; observed > model means people generalize from a lone example with a steeper magnitude gradient than the model's noisy baseline-gain similarity predicts (observed < model means a flatter gradient).
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0"), errors="coerce").fillna(0).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    sm = single.values
    dd = np.abs(t - x)
    a = sm & (dd >= 1) & (dd <= 10)
    b = sm & (dd > 30)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
