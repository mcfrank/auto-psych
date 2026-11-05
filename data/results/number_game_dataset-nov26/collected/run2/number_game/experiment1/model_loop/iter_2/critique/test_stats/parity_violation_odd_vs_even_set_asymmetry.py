# name: parity_violation_odd_vs_even_set_asymmetry
# description: Yes-rate for even targets when all examples are odd minus yes-rate for odd targets when all examples are even (multi-example sets only); observed > null means people tolerate parity violations of odd sets more than of even sets beyond what the model's single shared parity salience predicts, observed < null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    u = pd.unique(sets.values)
    par = {}
    for s in u:
        xs = [int(x) for x in s.split(",")]
        ps = {x % 2 for x in xs}
        par[s] = (list(ps)[0] if (len(ps) == 1 and len(xs) >= 2) else -1)
    p = sets.map(par).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    a = (p == 1) & (t % 2 == 0)
    b = (p == 0) & (t % 2 == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
