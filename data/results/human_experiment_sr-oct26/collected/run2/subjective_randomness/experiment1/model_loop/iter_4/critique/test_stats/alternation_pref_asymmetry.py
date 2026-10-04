# name: alternation_pref_asymmetry
# description: Rate of choosing the higher-switch-rate sequence when both rates are >= 0.5 minus the same rate when both are <= 0.5 (pairs with different rates only); observed below null_mean means people penalise over-alternation less sharply than the quadratic ideal implies, above means more sharply (i.e. the distance-to-ideal shape is wrong).
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    am = {s: alt(s) for s in u}
    a, b = df["sequence_a"].map(am).values, df["sequence_b"].map(am).values
    c = df["chose_left"].values
    chose_hi = np.where(a > b, c, 1 - c)
    diff = a != b
    hi = diff & (a >= 0.5) & (b >= 0.5)
    lo = diff & (a <= 0.5) & (b <= 0.5)
    rh = chose_hi[hi].mean() if hi.sum() else 0.5
    rl = chose_hi[lo].mean() if lo.sum() else 0.5
    return float(rh - rl)
