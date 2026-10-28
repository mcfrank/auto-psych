# name: shared_factor_nonmultiple_yes_rate
# description: In sets of 2+ examples whose common factor g is >= 3, yes-rate for targets that are not multiples of g; observed above the null means the model's abstracted multiples-of-g rule rejects non-multiples too strongly (people still endorse them by similarity), below means not strongly enough.
def test_statistic(df):
    sets = df["number_set"].astype(str).unique()
    gm = {}
    for s in sets:
        ex = [int(v) for v in s.split(",")]
        g = 0
        for v in ex:
            g = math.gcd(g, v)
        gm[s] = g if len(ex) >= 2 else 0
    g = df["number_set"].astype(str).map(gm).values.astype(int)
    t = df["target"].astype(int).values
    gs = np.where(g >= 3, g, 1)
    m = (g >= 3) & (t % gs != 0)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).values[m].mean())
