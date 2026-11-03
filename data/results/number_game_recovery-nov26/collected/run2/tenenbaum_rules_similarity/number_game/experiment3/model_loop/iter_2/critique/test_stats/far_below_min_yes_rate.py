# name: far_below_min_yes_rate
# description: Yes rate for targets below half the smallest example, among sets whose smallest example is at least 20; observed below null means people treat the smallest example as a floor (penalising downward extrapolation) which the model, with only an upward ceiling, under-penalises; above null the reverse.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).astype(float)
    mn = s.min(axis=1).values
    t = df["target"].astype(float).values
    m = (mn >= 20) & (t < mn / 2.0)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
