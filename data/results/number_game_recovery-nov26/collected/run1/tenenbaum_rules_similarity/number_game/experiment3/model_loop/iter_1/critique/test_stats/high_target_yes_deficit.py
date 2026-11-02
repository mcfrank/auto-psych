# name: high_target_yes_deficit
# description: Yes rate for targets 71-90 minus the yes rate for all other targets; observed below null means people endorse large numbers (70s-80s) less than the model predicts, a magnitude bias it lacks; above means the model under-endorses them.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = (t >= 71) & (t <= 90)
    if not m.any() or m.all():
        return 0.0
    return float(y[m].mean() - y[~m].mean())
