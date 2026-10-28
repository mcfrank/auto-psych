# name: last_step_direction_conflict
# description: In 3+-example sets whose last step (second-last to last) points opposite to the overall first-to-last direction, yes-rate for outside-span targets within 30 beyond the span in the last-step direction minus those within 30 in the first-to-last direction; observed above the null means people's momentum follows the most recent step rather than the first-to-last travel the model uses (below means the first-to-last direction dominates even more than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: [int(v) for v in s.split(",")] for s in sets}
    def ls(e):
        if len(e) < 3:
            return 0
        a = (e[-1] > e[0]) - (e[-1] < e[0]); b = (e[-1] > e[-2]) - (e[-1] < e[-2])
        return b if (a != 0 and b == -a) else 0
    lsd = ns.map({s: ls(e) for s, e in ex.items()}).values.astype(int)
    hi = ns.map({s: max(e) for s, e in ex.items()}).values
    lo = ns.map({s: min(e) for s, e in ex.items()}).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    above = (t > hi) & (t - hi <= 30)
    below = (t < lo) & (lo - t <= 30)
    with_last = ((lsd == 1) & above) | ((lsd == -1) & below)
    with_first = ((lsd == 1) & below) | ((lsd == -1) & above)
    if with_last.sum() == 0 or with_first.sum() == 0:
        return 0.0
    return float(y[with_last].mean() - y[with_first].mean())
